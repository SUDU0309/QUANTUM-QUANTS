import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load and clean training data
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorization and Model Training
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Load new dataset
df1 = pd.read_csv(r"C:\Users\HP\Downloads\extended_large_event_sentiment_dataset.csv")
df1 = df1.drop(columns=["Issue Type", "Sentiment", "Username"])

# Function for prediction and pie chart
def prediction_and_pie_chart(df1, model, vectorizer):
    Event_Name = input("Enter the event name: ").strip().lower()

    df1["Event Name"] = df1["Event Name"].str.strip().str.lower()
    matched_rows = df1[df1["Event Name"] == Event_Name]

    if matched_rows.empty:
        print(f"No comments found for event: {Event_Name}")
        return

    comments = matched_rows["Comment"].tolist()
    print(f"Found {len(comments)} comments for {Event_Name}.")

    comments_tfidf = vectorizer.transform(comments)
    predictions = model.predict(comments_tfidf)

    # Create DataFrame for results
    result_df = pd.DataFrame({"Comment": comments, "Sentiment": predictions})
    print(result_df)

    # Count sentiment types
    sentiment_counts = result_df["Sentiment"].value_counts()
    print("\nSentiment counts:")
    print(sentiment_counts)

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=["lightgreen", "gold", "salmon"])
    plt.title(f"Sentiment Distribution for '{Event_Name.title()}'")
    plt.ylabel("")  # Hide y-label for better look
    plt.show()

# Call function
prediction_and_pie_chart(df1, model, vectorizer)


