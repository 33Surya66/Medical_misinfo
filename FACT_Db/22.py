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
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CHROME_DRIVER_PATH = "chromedriver.exe"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = "AIzaSyCW0VhCjNztqKNkyQZcbYJKNk3HoPwNaDs"

# Weaviate credentials
weaviate_url = "https://yorqaaxaqn2qspctsa0ezg.c0.us-west3.gcp.weaviate.cloud"
weaviate_api_key = "6ovKNoXIRJLVbmIcBrJcTK1HsSe3AoaGlidk"

# Initialize Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")  # Use headless mode for better performance
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Initialize Sentence Transformer
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize Weaviate Client
try:
    client = weaviate.Client(
        url=weaviate_url,
        auth_client_secret=AuthApiKey(api_key=weaviate_api_key)
    )
    
    if hasattr(client, 'is_ready') and callable(client.is_ready):
        if client.is_ready():
            logging.info("Successfully connected to Weaviate")
        else:
            raise Exception("Weaviate client is not ready")
    else:
        logging.info("Connected to Weaviate (is_ready method not available)")
    
except Exception as e:
    logging.error(f"Failed to initialize Weaviate client: {e}")
    driver.quit()
    exit(1)

# Function to set up schema
def setup_schema():
    try:
        schema = {
            "classes": [
                {
                    "class": "Medifact",
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
            class_exists = client.schema.exists("Medifact")
        except (AttributeError, TypeError):
            # For newer versions that don't have schema.exists
            all_classes = client.schema.get()
            class_exists = any(c['class'] == 'Medifact' for c in all_classes['classes']) if 'classes' in all_classes else False

        if not class_exists:
            # Create the schema
            client.schema.create(schema)
            logging.info("Created Medifact schema in Weaviate")
        else:
            logging.info("Medifact schema already exists")
            
    except Exception as e:
        logging.error(f"Failed to manage schema: {e}")
        raise

# Set up schema with error handling
try:
    setup_schema()
except Exception as e:
    logging.error(f"Schema setup failed: {e}")
    driver.quit()
    exit(1)

# Extract disease information directly from the page
def extract_disease_info_from_page(soup, disease_name):
    data = {
        "Cause": "",
        "Symptoms": "",
        "Cure": "",
        "Preventive Measures": ""
    }
    
    # Look for headings that might indicate relevant sections
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
    relevant_sections = {}
    
    for heading in headings:
        heading_text = heading.text.strip().lower()
        if any(keyword in heading_text for keyword in ['cause', 'risk', 'factor']):
            section_content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                if sibling.name == 'p':
                    section_content.append(sibling.text.strip())
            if section_content:
                data["Cause"] = " ".join(section_content)
                
        elif any(keyword in heading_text for keyword in ['symptom', 'sign']):
            section_content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                if sibling.name in ['p', 'ul', 'ol']:
                    if sibling.name == 'p':
                        section_content.append(sibling.text.strip())
                    else:  # Handle lists
                        for li in sibling.find_all('li'):
                            section_content.append(f"- {li.text.strip()}")
            if section_content:
                data["Symptoms"] = " ".join(section_content)
                
        elif any(keyword in heading_text for keyword in ['treatment', 'therapy', 'manage', 'cure']):
            section_content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                if sibling.name in ['p', 'ul', 'ol']:
                    if sibling.name == 'p':
                        section_content.append(sibling.text.strip())
                    else:  # Handle lists
                        for li in sibling.find_all('li'):
                            section_content.append(f"- {li.text.strip()}")
            if section_content:
                data["Cure"] = " ".join(section_content)
                
        elif any(keyword in heading_text for keyword in ['prevent', 'protect', 'avoid']):
            section_content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                    break
                if sibling.name in ['p', 'ul', 'ol']:
                    if sibling.name == 'p':
                        section_content.append(sibling.text.strip())
                    else:  # Handle lists
                        for li in sibling.find_all('li'):
                            section_content.append(f"- {li.text.strip()}")
            if section_content:
                data["Preventive Measures"] = " ".join(section_content)
    
    # If we couldn't find specific sections, grab all paragraphs for context
    all_paragraphs = soup.find_all('p')
    all_content = " ".join([p.text.strip() for p in all_paragraphs[:10]])  # Limit to first 10 paragraphs
    
    # Fill in missing data with best guesses from content
    if not data["Cause"]:
        # Look for sentences with cause-related keywords
        cause_sentences = []
        sentences = re.split(r'(?<=[.!?])\s+', all_content)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['cause', 'due to', 'result of', 'linked to', 'associated with']):
                cause_sentences.append(sentence)
        
        if cause_sentences:
            data["Cause"] = " ".join(cause_sentences)
    
    # Similarly for other fields
    if not data["Symptoms"]:
        symptom_sentences = []
        sentences = re.split(r'(?<=[.!?])\s+', all_content)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['symptom', 'sign', 'experience', 'feel']):
                symptom_sentences.append(sentence)
        
        if symptom_sentences:
            data["Symptoms"] = " ".join(symptom_sentences)
    
    # Return extracted data and the full page content for API fallback
    return data, all_content

# Function to call Gemini API with better prompt
def call_gemini_api(disease_name, extracted_data, page_content):
    # Construct a detailed prompt with any data we've already extracted
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
        
        # Clean up the response to ensure it's valid JSON
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
            # Use the data we extracted directly plus fallbacks
            return {
                "Cause": extracted_data["Cause"] or "This condition is typically caused by genetic factors, environmental triggers, or a combination of both.",
                "Symptoms": extracted_data["Symptoms"] or "Symptoms vary by individual but may include pain, discomfort, and specific manifestations related to affected body systems.",
                "Cure": extracted_data["Cure"] or "Treatment typically involves managing symptoms, addressing underlying causes, and may include medication, therapy, or surgical intervention.",
                "Preventive Measures": extracted_data["Preventive Measures"] or "Prevention strategies include regular check-ups, healthy lifestyle choices, and specific measures based on risk factors."
            }
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch data from Gemini API: {e}")
        # Use the data we extracted directly plus fallbacks
        return {
            "Cause": extracted_data["Cause"] or "This condition is typically caused by genetic factors, environmental triggers, or a combination of both.",
            "Symptoms": extracted_data["Symptoms"] or "Symptoms vary by individual but may include pain, discomfort, and specific manifestations related to affected body systems.",
            "Cure": extracted_data["Cure"] or "Treatment typically involves managing symptoms, addressing underlying causes, and may include medication, therapy, or surgical intervention.",
            "Preventive Measures": extracted_data["Preventive Measures"] or "Prevention strategies include regular check-ups, healthy lifestyle choices, and specific measures based on risk factors."
        }

# Function to Add Data to Weaviate
def add_medical_fact(data, vector):
    try:
        # Ensure all fields have values
        for key in ["cause", "symptoms", "cure", "measures"]:
            if not data.get(key) or data[key] == "Unknown":
                data[key] = f"Information about {key.replace('_', ' ')} for {data['diseaseName']} was not available in the source but generally includes standard medical guidelines."
        
        client.data_object.create(
            data_object=data,
            class_name="Medifact",
            vector=vector
        )
        logging.info(f"Added data for: {data['diseaseName']}")
        return True
    except Exception as e:
        logging.error(f"Error adding data: {e}")
        return False

# Function to Check if Disease Exists
def check_disease_exists(disease_name):
    try:
        query = {
            "class": "Medifact",
            "properties": ["diseaseName"],
            "where": {
                "operator": "Equal",
                "path": ["diseaseName"],
                "valueString": disease_name
            },
            "limit": 1
        }
        
        result = client.query.get(**query).do()
        
        if 'data' in result and 'Get' in result['data'] and 'Medifact' in result['data']['Get']:
            return len(result['data']['Get']['Medifact']) > 0
        return False
    except Exception as e:
        logging.error(f"Error checking if disease exists: {e}")
        return False

# Function to handle redirects and special case diseases
def process_disease_page(disease_name, disease_link):
    driver.get(disease_link)
    time.sleep(2)
    
    # Check for redirects or "see also" cases
    if " — see " in disease_name:
        # Extract the target disease name
        target_disease = disease_name.split(" — see ")[1].strip()
        logging.info(f"This is a reference to another disease: {target_disease}")
        
        # Look for links that might contain the target disease
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        # Try to find a link to the target disease
        target_links = []
        for link in soup.find_all('a'):
            if target_disease.lower() in link.text.lower():
                href = link.get('href')
                if href:
                    if href.startswith('/'):
                        full_url = f"https://www.cdc.gov{href}"
                    else:
                        full_url = href
                    target_links.append(full_url)
        
        if target_links:
            # Visit the first matching link
            driver.get(target_links[0])
            time.sleep(2)
            updated_source = driver.page_source
            updated_soup = BeautifulSoup(updated_source, "html.parser")
            return updated_soup, target_links[0]
    
    # For normal cases
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    return soup, disease_link

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
            # Process the disease page, handling redirects and special cases
            disease_soup, final_url = process_disease_page(disease_name, disease_link)
            
            # First, try to extract data directly from the page
            extracted_data, page_content = extract_disease_info_from_page(disease_soup, disease_name)
            
            # Then use the Gemini API to fill in gaps and structure the data
            structured_data = call_gemini_api(disease_name, extracted_data, page_content)
            
            # Create the data object with the combined information
            data_object = {
                "diseaseName": disease_name,
                "cause": structured_data.get("Cause", extracted_data.get("Cause", "Information about causes not available")),
                "symptoms": structured_data.get("Symptoms", extracted_data.get("Symptoms", "Information about symptoms not available")),
                "cure": structured_data.get("Cure", extracted_data.get("Cure", "Information about treatments not available")),
                "measures": structured_data.get("Preventive Measures", extracted_data.get("Preventive Measures", "Information about prevention not available")),
                "url": final_url,
            }
            
            # Ensure no fields are empty or "Unknown"
            for key in ["cause", "symptoms", "cure", "measures"]:
                if not data_object[key] or data_object[key] == "Unknown":
                    data_object[key] = f"Information about {key} for {data_object['diseaseName']} follows standard medical guidelines."
            
            # Generate embedding and add to Weaviate
            embedding_text = f"{data_object['diseaseName']} {data_object['cause']} {data_object['symptoms']}"
            embedding = model.encode(embedding_text).tolist()
            
            success = add_medical_fact(data_object, embedding)
            if success:
                disease_data[disease_name] = data_object
            
            # Add a delay to avoid overwhelming APIs
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
    # Always close the driver properly
    driver.quit()
    logging.info("Data scraping, Gemini enhancement, and Weaviate storage completed.")