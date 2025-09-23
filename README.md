# LinkedIn Comment Intelligence API

A FastAPI application for analyzing LinkedIn comments with AI-powered insights, sentiment analysis, and data processing.

## 🚀 Features

- **AI-Powered Classification**: Automatic categorization of comments by themes and personas
- **Sentiment Analysis**: Advanced sentiment scoring and visualization
- **RAG (Retrieval-Augmented Generation)**: Intelligent comment search and insights
- **Persona Analysis**: Professional role classification (CEOs, Industry Leaders, Academic Staff, etc.)
- **Theme Categorization**: Comments grouped by Suggestions & Ideas, Support & Enthusiasm, Complementary Comments, and Negative Feedback
- **Engagement Metrics**: Analysis of replies, shares, and engagement patterns
- **REST API**: Easy integration with other applications

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
1. Clone the repository:
   ```bash
   git clone -b poc <repository-url>
   cd linkedin_analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API:
   ```bash
   python -m uvicorn app4_api:app --host 0.0.0.0 --port 8000 --reload
   ```

4. And then run the frontend (PostIn repository)