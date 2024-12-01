import requests
import csv
from bs4 import BeautifulSoup

# Create a User Agent
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Define Base URL
url = 'https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops'

# Send get() Request and fetch the webpage contents
response = requests.get(url, headers=headers)

# Create a Beautiful Soup Object
soup = BeautifulSoup(response.content, 'html.parser')

# Find all laptop items
laptop_items = soup.find_all('div', class_='col-md-4')

# Lists to store scraped data
names = []
prices = []
descriptions = []
ratings = []

# Extract information from each laptop item
for item in laptop_items:
    # Find title
    title_elem = item.find('a', class_='title')
    name = title_elem.text.strip() if title_elem else 'N/A'
    
    # Find price
    price_elem = item.find('h4', class_='pull-right')
    price = price_elem.text.strip() if price_elem else 'N/A'
    
    # Find description
    desc_elem = item.find('p', class_='description')
    description = desc_elem.text.strip() if desc_elem else 'N/A'
    
    # Find rating
    rating_elem = item.find('div', class_='ratings')
    rating_count = len(rating_elem.find_all('span', class_='glyphicon-star')) if rating_elem else 0
    
    # Append to lists
    names.append(name)
    prices.append(price)
    descriptions.append(description)
    ratings.append(rating_count)

# Explicitly write to CSV
try:
    # Open the CSV file in write mode with UTF-8 encoding
    with open('laptops_data.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
        # Create a CSV writer object
        csvwriter = csv.writer(csvfile)
        
        # Write the header
        csvwriter.writerow(['Title', 'Price', 'Description', 'Rating'])
        
        # Write the data rows
        for i in range(len(names)):
            csvwriter.writerow([
                names[i], 
                prices[i], 
                descriptions[i], 
                ratings[i]
            ])
    
    print(f"Successfully saved {len(names)} items to laptops_data.csv")

except Exception as e:
    print(f"An error occurred while writing to CSV: {e}")

# Optional: Print data to console for verification
print("\nScrapped Data:")
for i in range(len(names)):
    print(f"Title: {names[i]}")
    print(f"Price: {prices[i]}")
    print(f"Description: {descriptions[i]}")
    print(f"Rating: {ratings[i]}")
    print("---")