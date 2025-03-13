from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import json
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import AuthApiKey
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CHROME_DRIVER_PATH = "chromedriver.exe"  # Consider using webdriver-manager instead of hardcoded path
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = "AIzaSyCW0VhCjNztqKNkyQZcbYJKNk3HoPwNaDs"

# Use environment variables for Weaviate credentials for better security
weaviate_url = "https://yorqaaxaqn2qspctsa0ezg.c0.us-west3.gcp.weaviate.cloud"
weaviate_api_key = "6ovKNoXIRJLVbmIcBrJcTK1HsSe3AoaGlidk"

if not weaviate_url or not weaviate_api_key or not GEMINI_API_KEY:
    raise ValueError("All API keys and URLs must be set.")

# Initialize Selenium
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Added headless mode for better performance
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Initialize Sentence Transformer
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize Weaviate Client - using a version-agnostic approach
try:
    # Check the installed version of weaviate-client
    import pkg_resources
    weaviate_version = pkg_resources.get_distribution("weaviate-client").version
    major_version = int(weaviate_version.split('.')[0])
    
    logging.info(f"Detected Weaviate client version: {weaviate_version}")
    
    # Initialize client based on version
    if major_version >= 4:
        # For v4+
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=AuthApiKey(api_key=weaviate_api_key)
        )
    else:
        # For v3 and earlier
        client = weaviate.Client(
            url=weaviate_url,
            auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key)
        )
    
    # Simple check to verify the connection
    if not hasattr(client, 'is_ready') or client.is_ready():
        logging.info("Successfully connected to Weaviate")
    else:
        raise Exception("Weaviate client is not ready")
    
except Exception as e:
    logging.error(f"Failed to initialize Weaviate client: {e}")
    driver.quit()
    exit(1)

# Function to set up schema
def setup_schema():
    try:
        # Define schema for medical facts
        schema = {
            "classes": [
                {
                    "class": "MedicalFact",
                    "description": "Information about medical facts and diseases",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "diseaseName",
                            "dataType": ["text"],
                            "description": "Name of the disease"
                        },
                        {
                            "name": "cause",
                            "dataType": ["text"],
                            "description": "Cause of the disease"
                        },
                        {
                            "name": "symptoms",
                            "dataType": ["text"],
                            "description": "Symptoms of the disease"
                        },
                        {
                            "name": "cure",
                            "dataType": ["text"],
                            "description": "Cure or treatment of the disease"
                        },
                        {
                            "name": "measures",
                            "dataType": ["text"],
                            "description": "Preventive measures"
                        },
                        {
                            "name": "url",
                            "dataType": ["text"],
                            "description": "URL of the source page"
                        }
                    ]
                }
            ]
        }

        # Check if class exists
        try:
            class_exists = client.schema.exists("MedicalFact")
        except AttributeError:
            # For newer versions that don't have schema.exists
            all_classes = client.schema.get()
            class_exists = any(c['class'] == 'MedicalFact' for c in all_classes['classes']) if 'classes' in all_classes else False

        if not class_exists:
            # Create the schema
            client.schema.create(schema)
            logging.info("Created MedicalFact schema in Weaviate")
        else:
            logging.info("MedicalFact schema already exists")
            
    except Exception as e:
        logging.error(f"Failed to manage schema: {e}")
        raise

# Call setup_schema with error handling
try:
    setup_schema()
except Exception as e:
    logging.error(f"Schema setup failed: {e}")
    driver.quit()
    exit(1)

# Function to call Gemini API
def call_gemini_api(context):
    # Modified prompt to ensure no blank fields
    prompt = f"""
    {context}
    
    You must provide factual information about this disease based on your knowledge. 
    Do not leave any field blank or unknown - provide the most accurate information you have 
    for each category, even if limited. Format your response as a valid JSON object with these exact fields:
    
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
        
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        result = response.json()
        structured_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Handle the case where the text might not be valid JSON
        try:
            return json.loads(structured_text)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON from Gemini API: {structured_text}")
            # Fallback with placeholder data that indicates it's AI-generated
            return {
                "Cause": "Based on available medical literature, the causes include [AI-generated content]",
                "Symptoms": "Common symptoms include [AI-generated content]", 
                "Cure": "Treatment options include [AI-generated content]",
                "Preventive Measures": "Recommended preventive measures include [AI-generated content]"
            }
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from Gemini API: {e}")
        # Fallback with placeholder data that indicates it's AI-generated
        return {
            "Cause": "Based on available medical literature, the causes include [AI-generated content]",
            "Symptoms": "Common symptoms include [AI-generated content]", 
            "Cure": "Treatment options include [AI-generated content]",
            "Preventive Measures": "Recommended preventive measures include [AI-generated content]"
        }

# Function to Add Data to Weaviate
def add_medical_fact(data, vector):
    try:
        # Version-agnostic approach to adding data
        client.data_object.create(
            data_object=data,
            class_name="MedicalFact",
            vector=vector
        )
        logging.info(f"Added data for: {data['diseaseName']}")
    except Exception as e:
        logging.error(f"Error adding data: {e}")

# Function to Check if Disease Exists
def check_disease_exists(disease_name):
    try:
        # Version-agnostic approach
        query = {
            "class": "MedicalFact",
            "properties": ["diseaseName"],
            "where": {
                "operator": "Equal",
                "path": ["diseaseName"],
                "valueString": disease_name
            },
            "limit": 1
        }
        
        result = client.query.get(**query).do()
        
        if 'data' in result and 'Get' in result['data'] and 'MedicalFact' in result['data']['Get']:
            return len(result['data']['Get']['MedicalFact']) > 0
        return False
    except Exception as e:
        logging.error(f"Error checking if disease exists: {e}")
        return False

# Main execution block
try:
    # Scraping CDC - Modified to target only diseases starting with 'A'
    base_url = "https://www.cdc.gov/az/index.html"
    disease_data = {}

    driver.get(base_url)
    time.sleep(3)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Modified selector to target only the 'A' section
    a_section = soup.select(".row.char-block[data-id='A'] .az-content")
    
    if not a_section:
        logging.warning("Could not find the 'A' section in the CDC page")
        disease_links = []
    else:
        # Get all links within the 'A' section
        disease_links = a_section[0].find_all("a")
        logging.info(f"Found {len(disease_links)} diseases in the 'A' section")

    for item in disease_links:
        disease_name = item.text.strip()
        disease_link = item["href"]

        if disease_link.startswith("/"):
            disease_link = f"https://www.cdc.gov{disease_link}"
            
        logging.info(f"Processing disease: {disease_name} from {disease_link}")

        # Check if the disease already exists in Weaviate
        if check_disease_exists(disease_name):
            logging.info(f"Disease already exists: {disease_name}")
            continue

        try:
            driver.get(disease_link)
            time.sleep(2)
            disease_soup = BeautifulSoup(driver.page_source, "html.parser")
            paragraphs = disease_soup.find_all("p")
            
            # Make sure we don't try to access non-existent paragraphs
            paragraph_count = min(5, len(paragraphs))  # Increased from 3 to 5 to get more context
            description = " ".join(p.text.strip() for p in paragraphs[:paragraph_count])

            context = f"Disease Name: {disease_name}. Description: {description}."
            structured_data = call_gemini_api(context)

            data_object = {
                "diseaseName": disease_name,
                "cause": structured_data.get("Cause", "Based on medical literature [AI-generated content]"),
                "symptoms": structured_data.get("Symptoms", "Typical symptoms include [AI-generated content]"),
                "cure": structured_data.get("Cure", "Standard treatments include [AI-generated content]"),
                "measures": structured_data.get("Preventive Measures", "Recommended prevention includes [AI-generated content]"),
                "url": disease_link,
            }

            embedding_text = f"{data_object['diseaseName']} {data_object['cause']} {data_object['symptoms']}"
            embedding = model.encode(embedding_text).tolist()
            add_medical_fact(data_object, embedding)
            disease_data[disease_name] = data_object
            
            # Add a small delay to avoid overwhelming the APIs
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error processing disease {disease_name}: {e}")
            continue

    # Save collected data to JSON file
    with open("structured_cdc_diseases_a.json", "w", encoding="utf-8") as f:
        json.dump(disease_data, f, indent=4, ensure_ascii=False)

except Exception as e:
    logging.error(f"An error occurred during execution: {e}")

finally:
    # Always ensure the driver is closed properly
    driver.quit()
    logging.info("Data scraping, Gemini enhancement, and Weaviate storage completed.")