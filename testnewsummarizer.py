import ollama, os, json
from datetime import datetime, timedelta
from dotenv import load_dotenv; load_dotenv()
from Stage1DataCollector import Stage1DataCollector
from news_summarizer import NewsSummarizer

# Yesterday (Philippines UTC+8 = 1 day ahead of US markets)
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

print("Step 1: Loading cached articles...")
collector = Stage1DataCollector(api_key=os.getenv("BENZINGA_API"), cache_dir="data/cache")
articles  = collector.collect(yesterday)
total = sum(len(df) for df in articles.values())
print(f"  stock_news:    {len(articles['stock_news'])} articles")
print(f"  global_news:   {len(articles['global_news'])} articles")
print(f"  industry_news: {len(articles['industry_news'])} articles")
print(f"  total: {total}")

print()
print("Step 2: Sending to Ollama (qwen3:8b)...")

def llm_client(prompt):
    print(f"  Prompt length: {len(prompt)} chars")
    resp = ollama.chat(
        model="qwen3:14b",
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    # Support both old (dict) and new (object) ollama client versions
    content = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
    print(f"  Response length: {len(content)} chars")
    return content

summarizer = NewsSummarizer(llm_client=llm_client, window_days=7, verbose=True)
result = summarizer.summarize(articles, as_of_date=yesterday)

print()
print("Step 3: Result")
print("=" * 60)
print(json.dumps(result, indent=2))
