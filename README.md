# EMOTRIX - AI Sentiment Analysis Tool with Pattern Matching

This integrated application combines the beautiful EMOTRIX animation interface with sentiment analysis capabilities using pattern matching for detecting event-related issues.

## Requirements

- Python 3.6+
- Flask
- Pandas
- Matplotlib
- Re (Regular Expressions - built into Python)
- NumPy (for comparison charts)

## Installation

1. Make sure you have Python installed on your system.
2. Install the required packages:

```bash
pip install flask pandas matplotlib numpy
```

## Pattern-Matching Approach

The application uses simple but effective pattern matching techniques:
- **Sentiment Analysis**: Word-based patterns to identify positive, negative, and neutral sentiments
- **Issue Detection**: Specific patterns for detecting common event problems:
  - Crowd Management Issues
  - Time Delay Problems
  - Sound and Audio Issues
  - Technical Problems
  - Customer Service Issues
  - Venue Problems

## Data Files Required

The application needs one CSV file:
- `extended_large_event_sentiment_dataset.csv` - Contains the event data for analysis

The application is currently configured to load this file from:
- `C:\Users\HP\Downloads\extended_large_event_sentiment_dataset.csv`

If your file is in a different location, update the path in the `integrated_app.py` file.

## Running the Application

1. Navigate to the QUANTUM-QUANTS directory in your terminal
2. Run the integrated application:

```bash
python integrated_app.py
```

3. Open your web browser and go to `http://localhost:5000`
4. You'll see the EMOTRIX animation, then:
   - Click "Enter the Event" button
   - Select an event from the dropdown
   - Click "Analyze" to see sentiment analysis results

## Features

- Beautiful animation interface with smooth transitions and effects
- Pattern-based sentiment analysis
- Specialized issue detection focused on event-specific problems
- Confidence scores for both sentiment and issue detection
- Priority alert flagging for urgent negative comments
- Visual representation of sentiment distribution
- Detailed comment listing with sentiment indicators
- Fully scrollable interface with navigation controls
- Comment limiting to prevent performance issues (maximum 50 comments)

## Multi-Event Comparison

The application now includes a powerful comparison feature that lets you:
- Compare sentiment analysis results across multiple events (2-4 at a time)
- View comparative data in both line graphs and bar charts
- See percentage distributions of positive, negative, and neutral sentiments
- Toggle between different visualization styles
- View detailed statistics for each event side-by-side

To use the comparison feature:
1. First analyze any single event
2. Click on the "Compare with other events" link at the bottom of the results
3. Select 2-4 events using the checkboxes
4. Click "Compare Selected Events"
5. Toggle between Line Graph and Bar Chart views to see different visualizations

This feature is perfect for comparing sentiment trends across:
- Multiple company events
- Different product launches
- Competitive analysis between brands
- Historical analysis of recurring events

## Performance Considerations

To ensure optimal performance and responsiveness:
- Analysis is limited to a maximum of 50 comments per event
- The application will display a note when comments have been limited
- Issue detection is only applied to negative comments
- Comparison is limited to a maximum of 4 events at once

## Benefits of Pattern Matching

This pattern-matching approach offers several advantages:
- No dependencies on external ML libraries or models
- Faster processing compared to ML models
- No need for model downloads or internet connection
- Lightweight implementation suitable for all environments
- Easily customizable patterns for specific use cases
- Focus on event-specific issues like crowd management, time delays, and sound problems
- Transparent and explainable detection logic

All animations from the original interface have been preserved while providing focused issue detection. 