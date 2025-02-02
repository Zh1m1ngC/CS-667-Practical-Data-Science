import requests
from bs4 import BeautifulSoup
from newspaper import Article
from googlesearch import search
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

# Initialize models
sentiment_analyzer = SentimentIntensityAnalyzer()
bert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_article_text(url):
    """Extracts text content from the given URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

def check_citations(domain):
    """Searches Google Scholar for citations of the domain."""
    query = f"site:{domain}"
    results = list(search(query, num_results=5))
    return len(results)  # Number of Google Scholar citations

def check_relevance(user_query, content):
    """Computes relevance score using BERT similarity."""
    query_embedding = bert_model.encode(user_query, convert_to_tensor=True)
    content_embedding = bert_model.encode(content, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_embedding, content_embedding).item()
    return score

def check_bias(content):
    """Analyzes sentiment to detect extreme bias."""
    sentiment_score = sentiment_analyzer.polarity_scores(content)['compound']
    return abs(sentiment_score)  # Closer to 1 means more bias

def cross_verify(user_query):
    """Checks if multiple sources discuss the same topic."""
    similar_articles = list(search(user_query, num_results=5))
    return len(similar_articles)  # More sources mean more credibility

def rate_source(url, user_query):
    """Generates a validity score based on multiple factors."""
    domain = url.split("//")[-1].split("/")[0]
    content = get_article_text(url)

    if not content:
        return {"url": url, "validity_score": 0, "reason": "Failed to fetch content"}

    citation_score = min(check_citations(domain) / 10, 1)  # Normalize
    relevance_score = check_relevance(user_query, content)
    bias_score = 1 - check_bias(content)  # Lower bias = higher score
    cross_verification_score = min(cross_verify(user_query) / 5, 1)  # Normalize

    # Weighted Score Calculation
    validity_score = (
        0.3 * citation_score +
        0.25 * relevance_score +
        0.15 * bias_score +
        0.2 * cross_verification_score
    ) * 5  # Scale to 5-star rating

    return {
        "url": url,
        "validity_score": round(validity_score, 2),
        "details": {
            "citations": citation_score,
            "relevance": relevance_score,
            "bias": bias_score,
            "cross_verification": cross_verification_score,
        }
    }

# Example usage
user_query = "current trends in machine learning"
url = "https://www.techtarget.com/searchenterpriseai/tip/9-top-AI-and-machine-learning-trends"
print(rate_source(url, user_query))
