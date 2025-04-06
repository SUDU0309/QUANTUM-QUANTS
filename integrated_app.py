
    
    
# New implementation using Hugging Face free GPT models
import pandas as pd  # For data manipulation and analysis
import matplotlib  # For creating visualizations
# Set the backend to Agg (non-interactive) before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # For creating plots and charts
import json  # For handling JSON data
import io  # For handling input/output operations
import base64  # For encoding/decoding binary data
import re  # For regular expression operations
from flask import Flask, render_template, request, jsonify, Response  # Flask web framework components
import numpy as np  # For numerical operations
import requests  # For making HTTP requests
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical features
from sklearn.naive_bayes import MultinomialNB  # For the Naive Bayes classifier
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Initialize Flask application
app = Flask(__name__, template_folder='.')  # Create Flask app instance with current directory as template folder

# Function to initialize and train the machine learning model
def initialize_ml_model():
    """
    This function loads the training data, preprocesses it, and trains a sentiment analysis model.
    Steps:
    1. Load data from CSV file
    2. Clean and prepare the data
    3. Split into training and testing sets
    4. Convert text to numerical features
    5. Train the model
    """
    # Load the training dataset from CSV file
    df = pd.read_csv(r"C:\Users\HP\Downloads\sentiment_analysis_dataset.csv")
    
    # Remove unnecessary columns that won't be used for training
    df = df.drop(columns=["Event Name", "Platform", "Timestamp"])
    
    # Separate comments by sentiment for balanced training
    df_positive = df[df["Sentiment"] == "Positive"]  # Get all positive comments
    df_negative = df[df["Sentiment"] == "Negative"]  # Get all negative comments
    df_neutral = df[df["Sentiment"] == "Neutral"]   # Get all neutral comments
    
    # Combine all comments back together in a balanced way
    df_sorted = pd.concat([df_positive, df_neutral, df_negative], ignore_index=True)
    
    # Prepare features (X) and labels (y) for the model
    X = df_sorted["Text"]  # The actual comments (input features)
    y = df_sorted["Sentiment"]  # The sentiment labels (output to predict)
    
    # Split data into training (80%) and testing (20%) sets
    # random_state=42 ensures reproducibility
    # stratify=y ensures balanced distribution of sentiments in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize TF-IDF vectorizer to convert text to numerical features
    vectorizer = TfidfVectorizer()
    
    # Transform training text data into TF-IDF features
    # fit_transform learns the vocabulary and transforms the text
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Initialize Multinomial Naive Bayes classifier
    model = MultinomialNB()
    
    # Train the model on the transformed training data
    model.fit(X_train_tfidf, y_train)
    
    # Return both the trained model and vectorizer for later use
    return model, vectorizer

# Initialize the model and vectorizer when the application starts
model, vectorizer = initialize_ml_model()

# Load events data
events_df = pd.read_csv(r"C:\Users\HP\Downloads\extended_large_event_sentiment_dataset.csv")

# Define patterns for sentiment analysis
sentiment_patterns = {
    "Positive": [
        r"great", r"excellent", r"amazing", r"good", r"awesome", r"fantastic", 
        r"wonderful", r"love", r"enjoy", r"pleasant", r"impressed", r"happy", 
        r"perfect", r"satisfied", r"helpful", r"thank", r"best"
    ],
    "Negative": [
        r"bad", r"poor", r"terrible", r"awful", r"horrible", r"disappointing", 
        r"worst", r"hate", r"slow", r"issue", r"problem", r"not working", r"fail", 
        r"broken", r"difficult", r"expensive", r"waste", r"refund", r"angry", 
        r"not good", r"not happy", r"dissatisfied", r"complaint", r"crash", r"error"
    ],
    "Neutral": [
        r"ok", r"okay", r"fine", r"average", r"decent", r"normal", r"expected", 
        r"standard", r"adequat", r"reasonable", r"middle", r"fair", r"just"
    ]
}

# Define patterns for issue detection
issue_patterns = {
    "Crowd Management Issues": [
        r"crowd", r"overcrowd", r"too many people", r"too packed", r"no space", 
        r"cramped", r"packed", r"moving around", r"no room", r"stand", r"stampede", 
        r"push", r"shove", r"mob", r"jam", r"congestion", r"flow", r"volume of people"
    ],
    "Time Delay Problems": [
        r"delay", r"late", r"waited", r"waiting", r"start late", r"behind schedule", 
        r"overtime", r"long time", r"slow", r"queue", r"line", r"wast(ed|ing) time", 
        r"not on time", r"postpone", r"reschedule", r"arrived late", r"too long"
    ],
    "Sound and Audio Issues": [
        r"sound", r"audio", r"loud", r"volume", r"noise", r"couldn't hear", r"can't hear", 
        r"couldn't listen", r"echo", r"distort", r"speaker", r"microphone", r"quiet", 
        r"hearing", r"acoustic", r"feedback", r"static", r"crackling", r"too loud",
        r"not loud enough", r"music", r"bass", r"treble"
    ],
    "Technical Problems": [
        r"technical", r"glitch", r"bug", r"crash", r"error", r"malfunction", r"not working", 
        r"broken", r"failure", r"disconnect", r"connection", r"online", r"streaming", 
        r"video", r"app", r"website", r"system", r"outage", r"login", r"password"
    ],
    "Customer Service Issues": [
        r"service", r"staff", r"employee", r"rude", r"unhelpful", r"manager", 
        r"customer support", r"help desk", r"information", r"assist", r"attitude", 
        r"unprofessional", r"disrespectful", r"incompetent", r"untrained"
    ],
    "Venue Problems": [
        r"venue", r"facility", r"location", r"building", r"arena", r"stadium", 
        r"seating", r"chairs", r"uncomfortable", r"toilet", r"bathroom", r"restroom", 
        r"parking", r"temperature", r"hot", r"cold", r"air conditioning", r"ac", r"heat"
    ]
}

# Alert phrases that indicate high priority
alert_phrases = [
    "unacceptable", "fix this now", "worst experience", "report", 
    "legal action", "never use again", "disappointed", "very angry", 
    "scam", "uninstall", "refund", "terrible", "horrible", "awful",
    "demanding", "immediately", "urgently", "critical", "emergency"
]

# Function to analyze sentiment of a given text using the trained model
def analyze_sentiment(text):
    """
    This function analyzes the sentiment of a given text using the trained model.
    Steps:
    1. Convert text to numerical features
    2. Make prediction
    3. Calculate confidence score
    Returns:
    - sentiment: The predicted sentiment (Positive/Negative/Neutral)
    - confidence: How confident the model is in its prediction (0-1)
    """
    # Transform the input text into TF-IDF features using the trained vectorizer
    text_tfidf = vectorizer.transform([text])
    
    # Get the predicted sentiment class (Positive/Negative/Neutral)
    sentiment = model.predict(text_tfidf)[0]
    
    # Get the probability scores for each sentiment class
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Find the index of the predicted sentiment in the list of classes
    sentiment_index = list(model.classes_).index(sentiment)
    
    # Get the confidence score (probability) for the predicted sentiment
    confidence = probabilities[sentiment_index]
    
    # Return both the predicted sentiment and confidence score
    return sentiment, confidence

# Function to detect issues using pattern matching
def detect_issue(text):
    text_lower = text.lower()
    
    # Count matches for each issue type
    issue_scores = {issue: 0 for issue in issue_patterns}
    
    for issue, patterns in issue_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            issue_scores[issue] += len(matches)
    
    # Check for alert phrases
    alert_required = any(phrase in text_lower for phrase in alert_phrases)
    
    # Determine the primary issue
    if any(issue_scores.values()):
        best_issue = max(issue_scores.items(), key=lambda x: x[1])
        confidence = min(0.95, 0.6 + best_issue[1] * 0.1)  # Scale confidence based on match count
        return best_issue[0], alert_required, confidence
    else:
        return "Other Issues", alert_required, 0.5

# Function to analyze comments for a specific event
def analyze_event_comments(event_name):
    # Filter comments for the event
    events_df["Event Name"] = events_df["Event Name"].str.strip().str.lower()
    matched_rows = events_df[events_df["Event Name"] == event_name.strip().lower()]
    
    if matched_rows.empty:
        return None, 0
    
    comments = matched_rows["Comment"].tolist()
    
    # Process a maximum of 50 comments to prevent overloading
    if len(comments) > 50:
        comments = comments[:50]
    
    # Analyze sentiments using pattern matching
    sentiment_counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0
    }
    
    for comment in comments:
        sentiment, _ = analyze_sentiment(comment)
        sentiment_counts[sentiment] += 1
    
    return sentiment_counts, len(comments)

# Function to analyze issues for a specific event
def analyze_event_issues(event_name, include_comments=False):
    # Filter comments for the event
    events_df["Event Name"] = events_df["Event Name"].str.strip().str.lower()
    matched_rows = events_df[events_df["Event Name"] == event_name.strip().lower()]
    
    if matched_rows.empty:
        return None, 0
    
    comments = matched_rows["Comment"].tolist()
    
    # Analyze issues using pattern matching
    issue_counts = {issue: 0 for issue in issue_patterns}
    issue_counts["Other Issues"] = 0  # Add "Other Issues" to issue_counts
    
    issue_comments = {issue: [] for issue in issue_patterns}
    issue_comments["Other Issues"] = []  # Add "Other Issues" to issue_comments
    
    for comment in comments:
        # First check if the comment is negative
        sentiment, _ = analyze_sentiment(comment)
        if sentiment == "Negative":
            issue, alert, confidence = detect_issue(comment)
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            if include_comments:
                issue_comments[issue].append({
                    "text": comment,
                    "alert": alert,
                    "confidence": confidence
                })
    if include_comments:
        return issue_counts, len(comments), issue_comments
    else:
        return issue_counts, len(comments)

# Function to fetch YouTube comments
def fetch_youtube_comments(video_id, api_key):
    """
    Fetches comments from a YouTube video using the YouTube API.
    """
    try:
        # YouTube API URL for fetching comments
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults=50"
        
        response = requests.get(url)
        data = response.json()
        
        comments = []
        if "items" in data:
            for item in data["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": comment["textDisplay"],
                    "author": comment["authorDisplayName"]
                })
        return comments
    except Exception as e:
        print(f"Error fetching YouTube comments: {str(e)}")
        return []

# Function to fetch YouTube live stream comments
def fetch_live_chat_messages(video_id, api_key):
    """
    Fetches live chat messages from a YouTube live stream.
    """
    url = f"https://www.googleapis.com/youtube/v3/liveChat/messages?part=snippet&liveChatId={video_id}&key={api_key}&maxResults=50"
    response = requests.get(url)
    data = response.json()
    
    messages = []
    if "items" in data:
        for item in data["items"]:
            message = item["snippet"]
            messages.append({
                "text": message["displayMessage"],
                "author": message["authorChannelId"],
                "timestamp": message["publishedAt"]
            })
    return messages

# Function to fetch Instagram comments
def fetch_instagram_comments(post_url, access_token):
    """
    Fetches comments from an Instagram post using the Instagram Graph API.
    """
    try:
        # Extract the media ID from the post URL
        media_id = post_url.split('/p/')[1].split('/')[0]
        
        # Instagram Graph API endpoint for comments
        url = f"https://graph.instagram.com/{media_id}/comments"
        
        params = {
            'access_token': access_token,
            'fields': 'text,username,timestamp'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        comments = []
        if 'data' in data:
            for item in data['data']:
                comments.append({
                    "text": item.get('text', ''),
                    "author": item.get('username', ''),
                    "timestamp": item.get('timestamp', '')
                })
        return comments
    except Exception as e:
        print(f"Error fetching Instagram comments: {str(e)}")
        return []

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/get_events')
def get_events():
    events = events_df["Event Name"].unique().tolist()
    return jsonify({"events": events})

@app.route('/analyze', methods=['POST'])
def analyze_sentiment_endpoint():
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
    
    # Process a maximum of 50 comments to prevent overloading
    if len(comments) > 50:
        comments = comments[:50]
    
    # Analyze sentiments using pattern matching
    sentiment_results = []
    for comment in comments:
        sentiment, confidence = analyze_sentiment(comment)
        
        result = {
            "text": comment,
            "sentiment": sentiment,
            "confidence": confidence
        }
        
        # For negative comments, detect issues
        if sentiment == "Negative":
            issue, alert, issue_confidence = detect_issue(comment)
            result["issue"] = issue
            result["alert"] = alert
            result["issue_confidence"] = issue_confidence
            
        sentiment_results.append(result)
    
    # Count sentiments
    sentiment_counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0
    }
    
    for result in sentiment_results:
        sentiment_counts[result["sentiment"]] += 1
    
    # Define vibrant colors for the chart
    colors = {
        'Positive': '#00ff9d',  # Bright green
        'Neutral': '#ffdd00',   # Bright yellow
        'Negative': '#ff6d6d'   # Bright red
    }
    
    # Enhanced pie chart with better styling and effects
    plt.figure(figsize=(10, 8))
    plt.style.use('dark_background')
    
    # Create a darker background
    ax = plt.axes()
    ax.set_facecolor('#121212')
    
    # Create explode effect for the slices (slightly separated)
    explode = (0.02, 0.02, 0.02)
    
    # Create the pie chart with enhanced styling
    wedges, texts, autotexts = plt.pie(
        sentiment_counts.values(), 
        labels=sentiment_counts.keys(),
        autopct='%1.1f%%',
        explode=explode, 
        startangle=140, 
        colors=[colors['Positive'], colors['Neutral'], colors['Negative']],
        wedgeprops={
            'edgecolor': 'white',
            'linewidth': 1.5,
            'antialiased': True,
            'width': 0.6,  # Creates a donut chart effect
            'alpha': 0.9
        },
        shadow=True,
        radius=1
    )
    
    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    # Style the labels
    for text in texts:
        text.set_fontsize(14)
        text.set_color('white')
    
    # Add a circle at the center for donut chart effect
    center_circle = plt.Circle((0, 0), 0.3, fc='#121212', edgecolor='white', linewidth=1.5)
    ax.add_artist(center_circle)
    
    # Add title with enhanced styling
    plt.title(
        f"Sentiment Distribution for\n'{event_name.title()}'", 
        fontsize=18, 
        color='#7af7ff',
        fontweight='bold',
        pad=20
    )
    
    # Add a subtle glow effect around the pie chart
    for i, wedge in enumerate(wedges):
        path = wedge.get_path()
        vertices = path.vertices.copy()
        glow = plt.Polygon(
            vertices, 
            closed=True, 
            fill=True, 
            color=list(colors.values())[i],
            alpha=0.15,
            linewidth=0
        )
        ax.add_patch(glow)
    
    # Add a subtitle with total comment count
    plt.annotate(
        f"Based on {len(comments)} comments",
        xy=(0, -1.3),
        ha='center',
        va='center',
        fontsize=12,
        color='white',
        alpha=0.8,
        xycoords='axes fraction'
    )
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight', facecolor='#121212', edgecolor='none')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    # Prepare results
    results = {
        "sentiment_counts": sentiment_counts,
        "total_comments": len(comments),
        "chart": plot_url,
        "comments": sentiment_results,
        "note": "Limited to max 50 comments" if len(matched_rows) > 50 else ""
    }
    
    return jsonify(results)

@app.route('/compare', methods=['POST'])
def compare_events():
    data = request.json
    event_names = data.get('events', [])
    
    if not event_names or len(event_names) < 2:
        return jsonify({
            "error": "Please provide at least 2 events to compare"
        })
    
    if len(event_names) > 4:
        return jsonify({
            "error": "Maximum 4 events can be compared at once"
        })
    
    # Analyze each event
    all_results = {}
    total_comments = {}
    
    for event_name in event_names:
        sentiment_counts, comment_count = analyze_event_comments(event_name)
        if sentiment_counts:
            all_results[event_name] = sentiment_counts
            total_comments[event_name] = comment_count
        else:
            return jsonify({
                "error": f"No comments found for event: {event_name}"
            })
    
    # Define vibrant colors for the charts
    colors = {
        'Positive': ['#00ff9d', '#00c07a', '#00a268'], # Green shades
        'Neutral': ['#ffdd00', '#ffc107', '#ffaa00'],  # Yellow/Gold shades
        'Negative': ['#ff6d6d', '#ff4757', '#ff1f39']  # Red shades
    }
    
    # Enhanced bar chart with gradient fills and better styling
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    # Create a darker background
    ax = plt.axes()
    ax.set_facecolor('#121212')
    
    # Data for plotting
    categories = ["Positive", "Neutral", "Negative"]
    events = list(all_results.keys())
    x = np.arange(len(categories))
    width = 0.2  # Width of bars
    
    # Calculate percentages for each sentiment
    percentages = {}
    for event, counts in all_results.items():
        total = sum(counts.values())
        percentages[event] = [counts[cat]/total*100 for cat in categories]
    
    # Plot bars for each event with enhanced styling
    for i, (event, values) in enumerate(percentages.items()):
        # Create gradient colors for each bar
        for j, value in enumerate(values):
            # Get color for this category
            base_color = colors[categories[j]][i % len(colors[categories[j]])]
            
            # Plot bar with gradient effect
            bar = plt.bar(
                x[j] + (i - len(events)/2 + 0.5) * width, 
                values[j], 
                width, 
                label=event.title() if j == 0 else "",
                color=base_color,
                edgecolor='white',
                linewidth=0.5,
                alpha=0.8,
                zorder=3
            )
    
    # Add grid with custom styling
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#ffffff')
    
    # Add title and labels with enhanced styling
    plt.ylabel('Percentage (%)', fontsize=12, color='white')
    plt.title('Sentiment Comparison Across Events', fontsize=18, color='#7af7ff', fontweight='bold',
              pad=20, loc='center')
    
    # Make the tick labels more visible
    plt.xticks(x, categories, fontsize=12, color='white')
    plt.yticks(fontsize=10, color='white')
    
    # Adding a subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1)
    
    # Create a more visually appealing legend
    legend = plt.legend(title="Events", title_fontsize=11, fontsize=10, loc='upper right',
                         framealpha=0.7, edgecolor='#444444')
    legend.get_title().set_color('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add value labels on bars
    for i, (event, values) in enumerate(percentages.items()):
        for j, value in enumerate(values):
            plt.text(
                x[j] + (i - len(events)/2 + 0.5) * width, 
                value + 2, 
                f"{value:.1f}%", 
                ha='center', 
                va='bottom', 
                fontsize=9,
                color='white',
                fontweight='bold',
                alpha=0.9
            )
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight', facecolor='#121212', edgecolor='none')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    # Enhanced line graph for comparison with better styling
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    # Set dark background
    ax = plt.axes()
    ax.set_facecolor('#121212')
    
    # Plot line for each sentiment type with enhanced styling
    markers = ['o', 's', '^']
    
    # Add subtle background gradient
    plt.fill_between(
        np.arange(2), 
        [0, 0], 
        [100, 100], 
        color='#1a1a1a', 
        alpha=0.5
    )
    
    # Plot stylish grid lines
    plt.grid(True, linestyle='--', alpha=0.2, color='#ffffff')
    
    for i, category in enumerate(categories):
        values = [all_results[event][category] / total_comments[event] * 100 for event in events]
        
        # Plot main line
        plt.plot(
            range(len(events)), 
            values, 
            marker=markers[i], 
            label=category, 
            color=colors[category][1],
            linewidth=3,
            markersize=10,
            markeredgecolor='white',
            markeredgewidth=1,
            alpha=0.9,
            zorder=5
        )
        
        # Add a subtle glow effect
        plt.plot(
            range(len(events)), 
            values, 
            color=colors[category][1],
            linewidth=8,
            alpha=0.2,
            zorder=4
        )
        
        # Add value labels
        for j, value in enumerate(values):
            plt.text(
                j, 
                value + 2,
                f"{value:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9,
                color=colors[category][1],
                fontweight='bold',
                alpha=0.9,
                zorder=6
            )
    
    # Set custom styling for labels and title
    plt.ylabel('Percentage (%)', fontsize=12, color='white')
    plt.title('Sentiment Trends Across Events', fontsize=18, color='#7af7ff', fontweight='bold',
              pad=20, loc='center')
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1)
    
    # Make tick labels more readable
    plt.xticks(range(len(events)), events, rotation=45, ha='right', fontsize=11, color='white')
    plt.yticks(fontsize=10, color='white')
    
    # Enhanced legend
    legend = plt.legend(title="Sentiment Types", title_fontsize=11, fontsize=10, loc='upper right',
                         framealpha=0.7, edgecolor='#444444')
    legend.get_title().set_color('white')
    for text in legend.get_texts():
        text.set_color('white')
    
    plt.tight_layout()
    
    # Convert line graph to base64 string
    line_img = io.BytesIO()
    plt.savefig(line_img, format='png', dpi=120, bbox_inches='tight', facecolor='#121212', edgecolor='none')
    line_img.seek(0)
    line_plot_url = base64.b64encode(line_img.getvalue()).decode('utf8')
    plt.close()
    
    # Prepare response
    response = {
        "events": events,
        "sentiment_data": all_results,
        "total_comments": total_comments,
        "bar_chart": plot_url,
        "line_chart": line_plot_url
    }
    
    return jsonify(response)

@app.route('/issues_analysis', methods=['POST'])
def issues_analysis():
    data = request.json
    events = data.get('events', [])
    
    if not events:
        return jsonify({
            "error": "Please select at least one event to analyze"
        })
    
    # Analyze issues for each event
    all_results = {}
    total_comments = {}
    issue_details = {}
    
    for event_name in events:
        issue_counts, comment_count, issue_comments = analyze_event_issues(event_name, include_comments=True)
        if issue_counts:
            all_results[event_name] = issue_counts
            total_comments[event_name] = comment_count
            issue_details[event_name] = issue_comments
        else:
            return jsonify({
                "error": f"No comments found for event: {event_name}"
            })
    
    # Define vibrant colors for the chart
    issue_colors = {
        'Crowd Management Issues': '#ff6b6b',
        'Time Delay Problems': '#feca57',
        'Sound and Audio Issues': '#1dd1a1',
        'Technical Problems': '#5f27cd',
        'Customer Service Issues': '#54a0ff',
        'Venue Problems': '#ff9ff3',
        'Other Issues': '#576574'
    }
    
    # Prepare simple issue counts for display
    issue_counts = {}
    issues = list(issue_patterns.keys()) + ["Other Issues"]
    
    # Calculate totals across all events for each issue
    for issue in issues:
        issue_counts[issue] = {
            "total": 0,
            "events": {}
        }
        for event in events:
            count = all_results[event].get(issue, 0)
            if count > 0:
                issue_counts[issue]["total"] += count
                issue_counts[issue]["events"][event] = count
    
    # Sort issues by total count (descending)
    sorted_issues = sorted(issues, key=lambda x: issue_counts[x]["total"], reverse=True)
    
    # Prepare issue details for display - include comment text for each issue
    issue_comments_list = {}
    for issue in issues:
        issue_comments_list[issue] = []
        for event in events:
            if issue in issue_details[event]:
                for comment_data in issue_details[event][issue]:
                    issue_comments_list[issue].append({
                        "event": event,
                        "text": comment_data["text"],
                        "alert": comment_data["alert"]
                    })
    
    # Prepare response
    response = {
        "events": events,
        "issues": sorted_issues,
        "issue_counts": issue_counts,
        "issue_colors": issue_colors,
        "issue_comments": issue_comments_list,
        "total_comments": total_comments
    }
    
    return jsonify(response)

def analyze_youtube_comments(comments):
    """
    Analyzes sentiment of YouTube comments and generates visualization.
    """
    if not comments:
        return None, "No comments found for this video"
    
    # Analyze sentiments
    sentiment_results = []
    for comment in comments:
        sentiment, confidence = analyze_sentiment(comment["text"])
        
        result = {
            "text": comment["text"],
            "author": comment["author"],
            "sentiment": sentiment,
            "confidence": confidence
        }
        
        # For negative comments, detect issues
        if sentiment == "Negative":
            issue, alert, issue_confidence = detect_issue(comment["text"])
            result["issue"] = issue
            result["alert"] = alert
            result["issue_confidence"] = issue_confidence
            
        sentiment_results.append(result)
    
    # Count sentiments
    sentiment_counts = {
        "Positive": 0,
        "Neutral": 0,
        "Negative": 0
    }
    
    for result in sentiment_results:
        sentiment_counts[result["sentiment"]] += 1
    
    # Create visualization
    colors = {
        'Positive': '#00ff9d',  # Bright green
        'Neutral': '#ffdd00',   # Bright yellow
        'Negative': '#ff6d6d'   # Bright red
    }
    
    plt.figure(figsize=(10, 8))
    plt.style.use('dark_background')
    
    ax = plt.axes()
    ax.set_facecolor('#121212')
    
    explode = (0.02, 0.02, 0.02)
    
    wedges, texts, autotexts = plt.pie(
        sentiment_counts.values(), 
        labels=sentiment_counts.keys(),
        autopct='%1.1f%%',
        explode=explode, 
        startangle=140, 
        colors=[colors['Positive'], colors['Neutral'], colors['Negative']],
        wedgeprops={
            'edgecolor': 'white',
            'linewidth': 1.5,
            'antialiased': True,
            'width': 0.6,
            'alpha': 0.9
        },
        shadow=True,
        radius=1
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    for text in texts:
        text.set_fontsize(14)
        text.set_color('white')
    
    center_circle = plt.Circle((0, 0), 0.3, fc='#121212', edgecolor='white', linewidth=1.5)
    ax.add_artist(center_circle)
    
    plt.title(
        "YouTube Comments Sentiment Analysis", 
        fontsize=18, 
        color='#7af7ff',
        fontweight='bold',
        pad=20
    )
    
    plt.annotate(
        f"Based on {len(comments)} comments",
        xy=(0, -1.3),
        ha='center',
        va='center',
        fontsize=12,
        color='white',
        alpha=0.8,
        xycoords='axes fraction'
    )
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight', facecolor='#121212', edgecolor='none')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return {
        "sentiment_counts": sentiment_counts,
        "total_comments": len(comments),
        "chart": plot_url,
        "comments": sentiment_results,
        "note": "Limited to max 50 comments"
    }, None

@app.route('/youtube_analysis', methods=['POST'])
def youtube_analysis():
    data = request.json
    video_id = data.get('video_id')
    api_key = data.get('api_key')
    
    if not video_id or not api_key:
        return jsonify({
            "error": "Please provide both video ID and API key"
        })
    
    try:
        # Fetch comments from YouTube
        comments = fetch_youtube_comments(video_id, api_key)
        
        # Analyze comments and generate visualization
        results, error = analyze_youtube_comments(comments)
        
        if error:
            return jsonify({"error": error})
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "error": f"Error analyzing YouTube comments: {str(e)}"
        })

@app.route('/instagram_analysis', methods=['POST'])
def instagram_analysis():
    data = request.json
    post_url = data.get('post_url')
    access_token = data.get('access_token')
    
    if not post_url or not access_token:
        return jsonify({
            "error": "Please provide both post URL and access token"
        })
    
    try:
        # Fetch comments from Instagram
        comments = fetch_instagram_comments(post_url, access_token)
        
        if not comments:
            return jsonify({
                "error": "No comments found for this post"
            })
        
        # Analyze sentiments using pattern matching
        sentiment_results = []
        for comment in comments:
            sentiment, confidence = analyze_sentiment(comment["text"])
            
            result = {
                "text": comment["text"],
                "author": comment["author"],
                "sentiment": sentiment,
                "confidence": confidence,
                "timestamp": comment.get("timestamp", "")
            }
            
            # For negative comments, detect issues
            if sentiment == "Negative":
                issue, alert, issue_confidence = detect_issue(comment["text"])
                result["issue"] = issue
                result["alert"] = alert
                result["issue_confidence"] = issue_confidence
                
            sentiment_results.append(result)
        
        # Count sentiments
        sentiment_counts = {
            "Positive": 0,
            "Neutral": 0,
            "Negative": 0
        }
        
        for result in sentiment_results:
            sentiment_counts[result["sentiment"]] += 1
        
        # Define vibrant colors for the chart
        colors = {
            'Positive': '#00ff9d',  # Bright green
            'Neutral': '#ffdd00',   # Bright yellow
            'Negative': '#ff6d6d'   # Bright red
        }
        
        # Enhanced pie chart with better styling and effects
        plt.figure(figsize=(10, 8))
        plt.style.use('dark_background')
        
        # Create a darker background
        ax = plt.axes()
        ax.set_facecolor('#121212')
        
        # Create explode effect for the slices (slightly separated)
        explode = (0.02, 0.02, 0.02)
        
        # Create the pie chart with enhanced styling
        wedges, texts, autotexts = plt.pie(
            sentiment_counts.values(), 
            labels=sentiment_counts.keys(),
            autopct='%1.1f%%',
            explode=explode, 
            startangle=140, 
            colors=[colors['Positive'], colors['Neutral'], colors['Negative']],
            wedgeprops={
                'edgecolor': 'white',
                'linewidth': 1.5,
                'antialiased': True,
                'width': 0.6,  # Creates a donut chart effect
                'alpha': 0.9
            },
            shadow=True,
            radius=1
        )
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        # Style the labels
        for text in texts:
            text.set_fontsize(14)
            text.set_color('white')
        
        # Add a circle at the center for donut chart effect
        center_circle = plt.Circle((0, 0), 0.3, fc='#121212', edgecolor='white', linewidth=1.5)
        ax.add_artist(center_circle)
        
        # Add title with enhanced styling
        plt.title(
            "Instagram Post Comments Analysis", 
            fontsize=18, 
            color='#7af7ff',
            fontweight='bold',
            pad=20
        )
        
        # Add a subtitle with total comment count
        plt.annotate(
            f"Based on {len(comments)} comments",
            xy=(0, -1.3),
            ha='center',
            va='center',
            fontsize=12,
            color='white',
            alpha=0.8,
            xycoords='axes fraction'
        )
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=120, bbox_inches='tight', facecolor='#121212', edgecolor='none')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # Prepare results
        results = {
            "sentiment_counts": sentiment_counts,
            "total_comments": len(comments),
            "chart": plot_url,
            "comments": sentiment_results,
            "note": "Limited to max 50 comments"
        }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "error": f"Error analyzing Instagram comments: {str(e)}"
        })

@app.route('/analyze_comment', methods=['POST'])
def analyze_comment():
    data = request.json
    comment = data.get('comment')
    
    if not comment:
        return jsonify({
            "error": "Please provide a comment to analyze"
        })
    
    try:
        # Analyze sentiment
        sentiment, confidence = analyze_sentiment(comment)
        
        # Initialize result
        result = {
            "sentiment": sentiment,
            "confidence": confidence
        }
        
        # If negative, detect issues
        if sentiment == "Negative":
            issue, alert, issue_confidence = detect_issue(comment)
            result.update({
                "issue": issue,
                "alert": alert,
                "issue_confidence": issue_confidence
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": f"Error analyzing comment: {str(e)}"
        })

# Add a simple health check route
@app.route('/health')
def health_check():
    return "App is running with sentiment analysis"

# Override overflow settings if needed
@app.after_request
def add_header(response):
    if response.content_type.startswith('text/html'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')     