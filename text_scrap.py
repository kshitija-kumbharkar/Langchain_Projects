import requests
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

url = 'https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops'
response = requests.get(url, headers=headers)

print(f"Response Status Code: {response.status_code}")
if response.status_code != 200:
    print("Failed to fetch the webpage.")
    exit()

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

def save_to_file():
    try:
        with open('laptops_data.txt', 'w', encoding="utf-8") as f:
            # Write header
            f.write("LAPTOP SCRAPING RESULTS\n")
            f.write("=============================================\n\n")
            
            # Write each laptop's details
            for i in range(len(names)):
                f.write(f"Laptop #{i+1}\n")
                f.write(f"Title: {names[i]}\n")
                f.write(f"Price: {prices[i]}\n")
                f.write(f"Description: {descriptions[i]}\n")
                f.write(f"Rating: {ratings[i]}\n")
                f.write("\n=============================================\n\n")
        
        print("Data saved to 'laptops_data.txt'.")
        
    except Exception as e:
        print(f"An error occurred while writing to file: {e}")

# Save data to text file
save_to_file()

# Optional: Print data to console for verification
print("\nScrapped Data:")
for i in range(len(names)):
    print(f"Title: {names[i]}")
    print(f"Price: {prices[i]}")
    print(f"Description: {descriptions[i]}")
    print(f"Rating: {ratings[i]}")
    print("---")