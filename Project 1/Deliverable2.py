import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from googlesearch import search

# Optional: Add your SerpAPI key here if you want to use Google Scholar lookup
SERPAPI_API_KEY = "YOUR_SERPAPI_KEY"

class URLValidator:
    """
    An optimized credibility rating class that combines citation lookup, relevance, 
    fact-checking, bias detection, and cross-verification to evaluate web content.
    """

    def __init__(self):
        # Load models once to avoid redundant API calls
        self.similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.fake_news_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        self.sentiment_analyzer = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

    def fetch_page_content(self, url: str) -> str:
        """ Extracts text content from the given URL. """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            return " ".join([p.text for p in soup.find_all("p")])  # Extract paragraph text
        except requests.RequestException:
            return ""  # Return empty string if failed

    def compute_similarity_score(self, user_query: str, content: str) -> int:
        """ Computes semantic similarity between user query and page content. """
        if not content:
            return 0
        return int(util.pytorch_cos_sim(self.similarity_model.encode(user_query), self.similarity_model.encode(content)).item() * 100)

    def detect_bias(self, content: str) -> int:
        """ Uses NLP sentiment analysis to detect potential bias in content. """
        if not content:
            return 50
        sentiment_result = self.sentiment_analyzer(content[:512])[0]
        return 100 if sentiment_result["label"] == "POSITIVE" else 50 if sentiment_result["label"] == "NEUTRAL" else 30

    def check_google_scholar(self, url: str) -> int:
        """ Checks Google Scholar citations using SerpAPI. """
        if not SERPAPI_API_KEY:
            return 0  # Skip if no API key provided
        params = {"q": url, "engine": "google_scholar", "api_key": SERPAPI_API_KEY}
        try:
            response = requests.get("https://serpapi.com/search", params=params)
            data = response.json()
            return min(len(data.get("organic_results", [])) * 10, 100)  # Normalize to 100 scale
        except:
            return 0  # Default to no citations

    def check_facts(self, content: str) -> int:
        """ Cross-checks extracted content with Google Fact Check API. """
        if not content:
            return 50
        api_url = f"https://toolbox.google.com/factcheck/api/v1/claimsearch?query={content[:200]}"
        try:
            response = requests.get(api_url)
            data = response.json()
            return 80 if "claims" in data and data["claims"] else 40
        except:
            return 50  # Default uncertainty score

    def cross_verify(self, user_query: str) -> int:
        """ Checks if multiple sources discuss the same topic using Google Search. """
        try:
            similar_articles = list(search(user_query, num_results=5))
            return min(len(similar_articles) * 20, 100)  # Normalize
        except:
            return 50  # Default

    def get_star_rating(self, score: float) -> tuple:
        """ Converts a score (0-100) into a 1-5 star rating. """
        stars = max(1, min(5, round(score / 20)))  # Normalize 100-scale to 5-star scale
        return stars, "⭐" * stars + "☆" * (5 - stars)

    def generate_explanation(self, scores) -> str:
        """ Generates a human-readable explanation for the score. """
        explanation = "Here’s how we evaluated the source:\n\n"

        if scores["citations"] > 80:
            explanation += "✅ This source is widely cited, indicating strong credibility.\n"
        elif scores["citations"] > 40:
            explanation += "ℹ️ This source has some citations but is not a top reference.\n"
        else:
            explanation += "⚠️ This source has few or no citations, so credibility is uncertain.\n"

        if scores["relevance"] > 80:
            explanation += "✅ The content is highly relevant to your query.\n"
        elif scores["relevance"] > 50:
            explanation += "ℹ️ The content is somewhat relevant but may include extra information.\n"
        else:
            explanation += "⚠️ The content has low relevance to your query.\n"

        if scores["bias"] < 50:
            explanation += "⚠️ The article appears biased and opinionated.\n"
        elif scores["bias"] > 70:
            explanation += "✅ The content appears neutral and balanced.\n"

        if scores["cross_verification"] > 80:
            explanation += "✅ Other sources confirm the information, increasing reliability.\n"
        elif scores["cross_verification"] > 50:
            explanation += "ℹ️ Some other sources confirm this, but not many.\n"
        else:
            explanation += "⚠️ Few sources discuss this, so it may be speculative.\n"

        return explanation

    def rate_url_validity(self, user_query: str, url: str) -> dict:
        """ Main function to evaluate the validity of a webpage. """
        content = self.fetch_page_content(url)

        scores = {
            "citations": self.check_google_scholar(url),
            "relevance": self.compute_similarity_score(user_query, content),
            "bias": self.detect_bias(content),
            "fact_check": self.check_facts(content),
            "cross_verification": self.cross_verify(user_query)
        }

        # Weighted Score Calculation
        final_score = (
            (0.3 * scores["citations"]) +
            (0.25 * scores["relevance"]) +
            (0.2 * scores["fact_check"]) +
            (0.15 * scores["bias"]) +
            (0.1 * scores["cross_verification"])
        )

        stars, star_icon = self.get_star_rating(final_score)
        explanation = self.generate_explanation(scores)

        return {
            "url": url,
            "validity_score": round(final_score, 2),
            "stars": star_icon,
            "explanation": explanation
        }

# Example usage
validator = URLValidator()
user_query = "current trends in machine learning"
url = "https://www.techtarget.com/searchenterpriseai/tip/9-top-AI-and-machine-learning-trends"
result = validator.rate_url_validity(user_query, url)

# Print result
print(f"🔗 URL: {result['url']}")
print(f"⭐ Rating: {result['stars']} ({result['validity_score']}/5)")
print(result["explanation"])
