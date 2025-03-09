from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import json
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CHROME_DRIVER_PATH = "chromedriver.exe"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Use environment variables for API keys when possible, with fallbacks for development
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCW0VhCjNztqKNkyQZcbYJKNk3HoPwNaDs")
weaviate_url = os.environ.get("WEAVIATE_URL", "https://yorqaaxaqn2qspctsa0ezg.c0.us-west3.gcp.weaviate.cloud")
weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "6ovKNoXIRJLVbmIcBrJcTK1HsSe3AoaGlidk")

if not weaviate_url or not weaviate_api_key or not GEMINI_API_KEY:
    raise ValueError("All API keys and URLs must be set.")

# Initialize Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode for servers
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Initialize Sentence Transformer
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize Weaviate Client using the updated connection method
try:
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    # Verify the connection is successful
    if not client.is_ready():
        raise Exception("Weaviate client is not ready")

    logging.info("Successfully connected to Weaviate")

except Exception as e:
    logging.error(f"Failed to initialize Weaviate client: {e}")
    driver.quit()
    exit(1)

try:
    # Check if MedicalFact collection exists using the collections API
    all_collections = client.collections.list_all()
    
    collection_exists = "MedicalFact" in all_collections
    
    if not collection_exists:
        # Create a new collection
        collection = client.collections.create(
            name="MedicalFact",
            description="Information about medical facts and diseases",
            properties=[
                Property(name="diseaseName", data_type=DataType.TEXT, description="Name of the disease"),
                Property(name="cause", data_type=DataType.TEXT, description="Cause of the disease"),
                Property(name="symptoms", data_type=DataType.TEXT, description="Symptoms of the disease"),
                Property(name="cure", data_type=DataType.TEXT, description="Cure or treatment of the disease"),
                Property(name="measures", data_type=DataType.TEXT, description="Preventive measures"),
                Property(name="url", data_type=DataType.TEXT, description="URL of the source page"),
            ],
            vectorizer_config=None  # We'll provide our own vectors
        )
        logging.info("Created MedicalFact collection in Weaviate")
    else:
        # Get the existing collection
        collection = client.collections.get("MedicalFact")
        logging.info("MedicalFact collection already exists")

except Exception as e:
    logging.error(f"Failed to manage collections: {e}")
    driver.quit()
    exit(1)

# Function to call Gemini API
def call_gemini_api(disease_name, extracted_data, page_content):
    prompt = f"""
    I need factual information about the disease "{disease_name}".

    Here's what I've already extracted from a reliable source:

    Cause: {extracted_data['Cause'] if extracted_data['Cause'] else 'Not found in source.'}

    Symptoms: {extracted_data['Symptoms'] if extracted_data['Symptoms'] else 'Not found in source.'}

    Treatment/Cure: {extracted_data['Cure'] if extracted_data['Cure'] else 'Not found in source.'}

    Preventive Measures: {extracted_data['Preventive Measures'] if extracted_data['Preventive Measures'] else 'Not found in source.'}

    Additional context from the source:
    {page_content[:2000]}

    Based on the above information and your medical knowledge, please provide a comprehensive and factually correct entry for this disease.

    You must provide accurate information for all fields, never leaving any field blank.
    If information for a field is not explicitly provided in the source, use your medical knowledge to provide accurate general information about this disease.

    Format your response as a valid JSON object with these exact fields:

    {{
      "Cause": "detailed cause information",
      "Symptoms": "comprehensive list of symptoms", 
      "Cure": "available treatments and cures",
      "Preventive Measures": "ways to prevent this disease"
    }}

    Return ONLY the JSON object with NO additional text.
    """

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        response.raise_for_status()

        result = response.json()
        structured_text = result['candidates'][0]['content']['parts'][0]['text']

        structured_text = structured_text.strip()
        if structured_text.startswith("```json"):
            structured_text = structured_text[7:]
        if structured_text.endswith("```"):
            structured_text = structured_text[:-3]
        structured_text = structured_text.strip()

        try:
            return json.loads(structured_text)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from Gemini API: {structured_text}")
            return {
                "Cause": extracted_data["Cause"] or "This condition is typically caused by genetic factors, environmental triggers, or a combination of both.",
                "Symptoms": extracted_data["Symptoms"] or "Symptoms vary by individual but may include pain, discomfort, and specific manifestations related to affected body systems.",
                "Cure": extracted_data["Cure"] or "Treatment typically involves managing symptoms, addressing underlying causes, and may include medication, therapy, or surgical intervention.",
                "Preventive Measures": extracted_data["Preventive Measures"] or "Prevention strategies include regular check-ups, healthy lifestyle choices, and specific measures based on risk factors."
            }

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from Gemini API: {e}")
        return {
            "Cause": extracted_data["Cause"] or "This condition is typically caused by genetic factors, environmental triggers, or a combination of both.",
            "Symptoms": extracted_data["Symptoms"] or "Symptoms vary by individual but may include pain, discomfort, and specific manifestations related to affected body systems.",
            "Cure": extracted_data["Cure"] or "Treatment typically involves managing symptoms, addressing underlying causes, and may include medication, therapy, or surgical intervention.",
            "Preventive Measures": extracted_data["Preventive Measures"] or "Prevention strategies include regular check-ups, healthy lifestyle choices, and specific measures based on risk factors."
        }

# Function to Add Data to Weaviate
def add_medical_fact(data, vector):
    try:
        collection = client.collections.get("MedicalFact")
        
        # Insert data with vector
        result = collection.data.insert(
            properties=data,
            vector=vector
        )
        
        logging.info(f"Added data for: {data['diseaseName']}")
        return result
    except Exception as e:
        logging.error(f"Error adding data: {e}")
        return None

# Function to Check if Disease Exists
def check_disease_exists(disease_name):
    try:
        collection = client.collections.get("MedicalFact")
        
        # Using the newer Filter API
        from weaviate.classes.query import Filter
        
        response = collection.query.fetch_objects(
            limit=1,
            filters=Filter.by_property("diseaseName").equal(disease_name)
        )
        
        return len(response.objects) > 0
    except Exception as e:
        logging.error(f"Error checking if disease exists: {e}")
        return False

# Main execution block
try:
    base_url = "https://www.cdc.gov/az/index.html"
    disease_data = {}

    driver.get(base_url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    disease_links = soup.select(".row.char-block a")
    
    # For testing with a smaller dataset, uncomment this:
    # disease_links = disease_links[:5]  # Process only first 5 diseases
    
    for item in disease_links:
        disease_name = item.text.strip()
        disease_link = item["href"]

        if disease_link.startswith("/"):
            disease_link = f"https://www.cdc.gov{disease_link}"

        if check_disease_exists(disease_name):
            logging.info(f"Disease already exists: {disease_name}")
            continue

        try:
            driver.get(disease_link)
            time.sleep(2)
            disease_soup = BeautifulSoup(driver.page_source, "html.parser")
            paragraphs = disease_soup.find_all("p")

            paragraph_count = min(3, len(paragraphs))
            description = " ".join(p.text.strip() for p in paragraphs[:paragraph_count])
            page_content = disease_soup.get_text()

            extracted_data = {
                "Cause": "",
                "Symptoms": "",
                "Cure": "",
                "Preventive Measures": "",
            }

            sections = disease_soup.find_all(['h2', 'h3', 'h4'])
            section_data = {}

            for section in sections:
                section_title = section.text.strip().lower()
                section_content = ""

                next_elem = section.next_sibling
                while next_elem and next_elem.name not in ['h2', 'h3', 'h4']:
                    if hasattr(next_elem, 'name') and next_elem.name == 'p':
                        section_content += " " + next_elem.text.strip()
                    next_elem = next_elem.next_sibling

                if section_content:
                    section_data[section_title] = section_content

            for key in ['cause', 'symptom', 'treatment', 'prevention']:
                for section_title, content in section_data.items():
                    if key in section_title:
                        if key == 'cause':
                            extracted_data["Cause"] += f" {section_title.capitalize()}: {content}"
                        elif key == 'symptom':
                            extracted_data["Symptoms"] += f" {section_title.capitalize()}: {content}"
                        elif key == 'treatment':
                            extracted_data["Cure"] += f" {section_title.capitalize()}: {content}"
                        elif key == 'prevention':
                            extracted_data["Preventive Measures"] += f" {section_title.capitalize()}: {content}"

            structured_data = call_gemini_api(disease_name, extracted_data, page_content)

            data_object = {
                "diseaseName": disease_name,
                "cause": structured_data.get("Cause", "Unknown"),
                "symptoms": structured_data.get("Symptoms", "Unknown"),
                "cure": structured_data.get("Cure", "Unknown"),
                "measures": structured_data.get("Preventive Measures", "Unknown"),
                "url": disease_link,
            }

            embedding_text = f"{data_object['diseaseName']} {data_object['cause']} {data_object['symptoms']}"
            embedding = model.encode(embedding_text).tolist()
            add_medical_fact(data_object, embedding)
            disease_data[disease_name] = data_object
            
            # Add a small delay between requests to avoid overwhelming servers
            time.sleep(1)

        except Exception as e:
            logging.error(f"Error processing disease {disease_name}: {e}")
            continue

    with open("structured_cdc_diseases.json", "w", encoding="utf-8") as f:
        json.dump(disease_data, f, indent=4, ensure_ascii=False)

except Exception as e:
    logging.error(f"An error occurred during execution: {e}")

finally:
    driver.quit()
    logging.info("Data scraping, Gemini enhancement, and Weaviate storage completed.")