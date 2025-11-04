import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import requests
from dotenv import load_dotenv
import os

# --- Load API key from .env ---
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# --- Ask for topic ---
topic = input("Enter the topic here: ").strip()

# --- NewsAPI endpoint with full fields ---
url = (
    f"https://newsapi.org/v2/everything?"
    f"q={topic}&"
    f"language=en&"
    f"sortBy=publishedAt&"
    f"pageSize=50&"  # max 100 for free tier
    f"apiKey={API_KEY}"
)

# --- Fetch data ---
response = requests.get(url)
data = response.json()

if data.get("status") != "ok":
    print("❌ API Error:", data.get("message", "Unknown error"))
    exit()

articles = data.get("articles", [])
if not articles:
    print("⚠️ No news articles found for this topic.")
    exit()

# --- Extract all details ---
records = []
for a in articles:
    records.append({
        "source": a["source"]["name"] if a.get("source") else None,
        "author": a.get("author"),
        "title": a.get("title"),
        "description": a.get("description"),
        "url": a.get("url"),
        "publishedAt": a.get("publishedAt"),
        "content": a.get("content")
    })

df = pd.DataFrame(records)
print(f"✅ {len(df)} articles fetched for '{topic}'")

# --- Clean text for sentiment ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

df["clean_text"] = df["title"].apply(clean_text)

# --- Run sentiment analysis on titles ---
print("\n🧠 Running sentiment analysis...")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
df["sentiment"] = df["clean_text"].apply(lambda x: sentiment_pipeline(x[:512])[0]["label"] if x else "NEUTRAL")

# --- Sentiment summary ---
sentiment_counts = df["sentiment"].value_counts()
print("\n📊 Sentiment summary:")
print(sentiment_counts)

# --- Visualization ---
plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=140)
plt.title(f"Public Sentiment on '{topic}'")
plt.show(block=False)

# --- Save full dataset ---
os.makedirs("data", exist_ok=True)
output_file = f"data/news_sentiment_full_{topic.replace(' ', '_')}.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"\n✅ Full results (with all NewsAPI details) saved as: {output_file}")
