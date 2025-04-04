from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

app = Flask(__name__)

# Load and prepare the model (this would normally be in a separate module)
def initialize_model():
    # Load training data
    df = pd.read_csv(r"C:\Users\HP\Downloads\sentiment_analysis_dataset.csv")
    df = df.drop(columns=["Event Name", "Platform", "Timestamp"])
    
    # Sort sentiments
    df_positive = df[df["Sentiment"] == "Positive"]
    df_negative = df[df["Sentiment"] == "Negative"]
    df_neutral = df[df["Sentiment"] == "Neutral"]
    df_sorted = pd.concat([df_positive, df_neutral, df_negative], ignore_index=True)
    
    # Prepare data
    X = df_sorted["Text"]
    y = df_sorted["Sentiment"]
    
    # Train model
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_tfidf, y)
    
    return model, vectorizer

# Initialize model and vectorizer
model, vectorizer = initialize_model()

# Load events data
events_df = pd.read_csv(r"C:\Users\HP\Downloads\extended_large_event_sentiment_dataset.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_events')
def get_events():
    events = events_df["Event Name"].unique().tolist()
    return jsonify({"events": events})

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    event_name = data.get('event').strip().lower()
    
    # Filter comments for the event
    events_df["Event Name"] = events_df["Event Name"].str.strip().str.lower()
    matched_rows = events_df[events_df["Event Name"] == event_name]
    
    if matched_rows.empty:
        return jsonify({
            "error": f"No comments found for event: {event_name}"
        })
    
    comments = matched_rows["Comment"].tolist()
    
    # Predict sentiments
    comments_tfidf = vectorizer.transform(comments)
    predictions = model.predict(comments_tfidf)
    
    # Count sentiments
    sentiment_counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0
    }
    
    for sentiment in predictions:
        sentiment_counts[sentiment] += 1
    
    # Prepare results
    results = {
        "sentiment_counts": sentiment_counts,
        "total_comments": len(comments),
        "comments": [{"text": comment, "sentiment": sentiment} 
                    for comment, sentiment in zip(comments, predictions)]
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True) 