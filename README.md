# EMOTRIX - Sentiment Analysis Web Application

EMOTRIX is a web application that combines beautiful animations with powerful sentiment analysis capabilities. It analyzes comments and feedback for various events and provides visual insights into the sentiment distribution.

## Features

- Interactive web interface with modern design
- Real-time sentiment analysis of event comments
- Visual representation of sentiment distribution using charts
- Detailed view of individual comments with sentiment labels
- Responsive and user-friendly design

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Required datasets:
  - `sentiment_analysis_dataset.csv` (for model training)
  - `extended_large_event_sentiment_dataset.csv` (for event data)

## Installation

1. Clone this repository or download the source code.

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure your dataset files are in the correct location:
   - `sentiment_analysis_dataset.csv`
   - `extended_large_event_sentiment_dataset.csv`

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

4. Select an event from the dropdown menu and click "Analyze Sentiments" to see the results.

## Project Structure

- `app.py` - Main Flask application file
- `templates/index.html` - Frontend template
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Technologies Used

- Backend:
  - Flask (Python web framework)
  - scikit-learn (Machine learning library)
  - pandas (Data manipulation)
  - NumPy (Numerical computing)

- Frontend:
  - HTML5
  - CSS3
  - JavaScript
  - Chart.js (Visualization library)

## License

This project is open source and available under the MIT License. 