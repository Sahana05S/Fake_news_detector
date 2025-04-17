import requests

API_KEY = "aea23b03e36f40afabc9adcbe158ba4d"  # replace this with your NewsAPI key
FLASK_API_URL = "http://127.0.0.1:5000/predict"

# Step 1: Get top headlines
news_url = f"https://newsapi.org/v2/top-headlines?country=in&pageSize=5&apiKey={API_KEY}"
response = requests.get(news_url)
data = response.json()

if data["status"] != "ok":
    print("Failed to fetch news:", data)
else:
    articles = data["articles"]

    for i, article in enumerate(articles, start=1):
        title = article["title"] or ""
        content = article["description"] or ""  # use description if content is None

        combined_text = f"{title}\n{content}"
        print(f"\n📰 News #{i}: {title}")

        # Step 2: Send to your ML model via Flask
        res = requests.post(FLASK_API_URL, json={
            "text": combined_text,
            "model": "isot"  # or "liar"
        })

        if res.status_code == 200:
            result = res.json()
            print(f"✅ Prediction: {result['result']} (Model: {result['model_used']})")
        else:
            print("❌ Error from model:", res.text)
