import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function: Detect issue and alert from comment
def detect_issue_and_alert(text):
    text = text.lower()

    issue_patterns = {
        "Connectivity": [
            r"(no|poor|bad|lost|disconnect(ed)?|drop(ping)?) (internet|network|wifi|signal)",
            r"(internet|wifi|network) (is )?(down|slow|unstable|not working|dead)"
        ],
        "Payment": [
            r"(failed|declined|error|problem) (in )?(payment|transaction|checkout)",
            r"(not|never|didn’t|did not) (receive|get) (refund|money|amount)"
        ],
        "Performance": [
            r"(too )?(slow|lag|freeze|hang|crash|unresponsive|delay|buggy)",
            r"(loading|opening|startup) (time|speed|delay|issue)"
        ],
        "Customer Service": [
            r"(no|poor|bad) (customer|support|service|help)",
            r"(tried|called|emailed) (support|help) (but|and)? (no|didn’t|not)? (response|reply|answer)"
        ],
        "App": [
            r"(app|application|mobile|interface) (is )?(crashing|buggy|unusable|not working|failing)",
            r"can’t (install|update|open|launch) (the )?(app|application)"
        ]
    }

    alert_phrases = [
        "this is unacceptable",
        "fix this now",
        "worst experience",
        "report this",
        "take legal action",
        "never use this again",
        "disappointed",
        "very angry",
        "this is a scam",
        "will uninstall"
    ]

    issue_scores = defaultdict(int)
    alert_required = False

    for issue, patterns in issue_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                issue_scores[issue] += len(matches)

    for phrase in alert_phrases:
        if phrase in text:
            alert_required = True
            break

    detected_issue = max(issue_scores, key=issue_scores.get) if issue_scores else "Other"
    return detected_issue, alert_required

# ---------------------------
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

# ---------------------------
# Load new dataset for prediction
df1 = pd.read_csv(r"C:\Users\HP\Downloads\extended_large_event_sentiment_dataset.csv")
df1 = df1.drop(columns=["Issue Type", "Sentiment", "Username"])

# ---------------------------
# Final prediction function
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

    # Create result DataFrame
    result_df = pd.DataFrame({"Comment": comments, "Sentiment": predictions})

    # Detect issues and alerts
    def analyze_row(row):
        if row["Sentiment"] == "Negative":
            issue, alert = detect_issue_and_alert(row["Comment"])
            return pd.Series([issue, alert])
        else:
            return pd.Series(["", False])

    result_df[["Detected Issue", "Alert"]] = result_df.apply(analyze_row, axis=1)

    print("\nFull Analysis:")
    print(result_df)

    # Sentiment count & chart
    sentiment_counts = result_df["Sentiment"].value_counts()
    print("\nSentiment counts:")
    print(sentiment_counts)

    # Pie Chart
    plt.figure(figsize=(6, 6))
    sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=["lightgreen", "gold", "salmon"])
    plt.title(f"Sentiment Distribution for '{Event_Name.title()}'")
    plt.ylabel("")
    plt.show()

# Call the function
prediction_and_pie_chart(df1, model, vectorizer)
