from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json

# Set up Selenium WebDriver with options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_driver_path = "chromedriver.exe"
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Base URL for CDC A-Z disease list
base_url = "https://www.cdc.gov/health-topics.html#A"

# Dictionary to store disease data
disease_data = {}

# Load the main CDC page
driver.get(base_url)
time.sleep(3)  # Allow time for the page to load

# Get page source and parse with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find all disease links inside the ".az-content" section
disease_list = soup.select(".az-content a")

for item in disease_list:
    disease_name = item.text.strip()
    disease_link = item["href"]

    # Convert relative links to absolute URLs
    if disease_link.startswith("/"):
        disease_link = f"https://www.cdc.gov{disease_link}"

    # Visit each disease page
    driver.get(disease_link)
    time.sleep(2)
    disease_soup = BeautifulSoup(driver.page_source, "html.parser")

    # Extract all paragraphs as a brief description
    paragraphs = disease_soup.find_all("p")
    description = " ".join(p.text.strip() for p in paragraphs[:3]) if paragraphs else "No description available"

    # Store data
    disease_data[disease_name] = {
        "url": disease_link,
        "description": description
    }

# Close Selenium WebDriver
driver.quit()

# Save data to JSON file
with open("cdc_diseases.json", "w", encoding="utf-8") as f:
    json.dump(disease_data, f, indent=4, ensure_ascii=False)

print("✅ Scraping complete! Data saved in 'cdc_diseases.json'.")
