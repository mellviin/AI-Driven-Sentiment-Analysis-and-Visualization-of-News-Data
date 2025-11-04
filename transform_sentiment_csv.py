import pandas as pd
import os
from datetime import datetime

# --- File paths ---
input_file = "data/news_sentiment_full_Hiring_in_India.csv"
output_file = "data/news_sentiment_transformed.csv"

if not os.path.exists(input_file):
    print(f"❌ File not found: {input_file}")
    exit()

# --- Step 1: Load CSV ---
df = pd.read_csv(input_file)
print(f"✅ Loaded {len(df)} rows")

# --- Step 2: Clean missing data ---
df.fillna("", inplace=True)

# --- Step 3: Normalize Sentiment Labels ---
def normalize_sentiment(text):
    text = str(text).strip().lower()
    if "pos" in text:
        return "Positive"
    elif "neg" in text:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment_Label"] = df["sentiment"].apply(normalize_sentiment)

# --- Step 4: Add Sentiment Score ---
def sentiment_score(label):
    if label == "Positive":
        return 1
    elif label == "Negative":
        return -1
    else:
        return 0

df["Sentiment_Score"] = df["Sentiment_Label"].apply(sentiment_score)

# --- Step 5: Extract and format date/time ---
df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
df["Date"] = df["publishedAt"].dt.date
df["Time"] = df["publishedAt"].dt.time

# --- Step 6: Select and reorder columns for Power BI ---
final_df = df[[
    "Date", "Time", "source", "author", "title", "description",
    "sentiment", "Sentiment_Label", "Sentiment_Score", "url", "content"
]]

# --- Step 7: Save the transformed file ---
os.makedirs("data", exist_ok=True)
final_df.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Transformed file saved as: {output_file}")

# --- Preview ---
print(final_df.head(5))
