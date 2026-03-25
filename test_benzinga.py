import requests
import json

def get_benzinga_news():
    # Replace with your actual API key
    api_key = "YOUR_TOKEN_HERE" 
    
    # Benzinga News API endpoint
    url = "https://api.benzinga.com/api/v2/news"
    
    # Setup the query parameters
    params = {
        "token": api_key,
        "pageSize": 5,           # Limit the results to 5 articles
        "displayOutput": "full"  # Get the full data format
    }
    
    # Setup headers as requested by the documentation
    headers = {
        "accept": "application/json"
    }

    print("Fetching news from Benzinga...\n")
    response = requests.get(url, params=params, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        news_items = response.json()
        
        # Iterate through the first 5 articles
        for index, article in enumerate(news_items[:5], start=1):
            title = article.get('title', 'No Title')
            date = article.get('created_at', 'No Date')
            url = article.get('url', 'No URL')
            author = article.get('author', 'Unknown Author')
            
            print(f"{index}. {title}")
            print(f"   Author: {author} | Date: {date}")
            print(f"   Read more: {url}")
            print("-" * 50)
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        print("Response:", response.text)

if __name__ == "__main__":
    get_benzinga_news()