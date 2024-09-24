import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":

            query = parse_qs(environ.get('QUERY_STRING', ''))
            filtered_reviews = reviews

            for review in filtered_reviews:
                if 'sentiment' not in review:
                    review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            location = query.get('location', [None])[0]
            if location:
                filtered_reviews = [review for review in filtered_reviews if review.get('Location') == location]

            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]
            if start_date:
                filtered_reviews = [review for review in filtered_reviews if review.get('Timestamp') >= start_date]
            if end_date:
                filtered_reviews = [review for review in filtered_reviews if review.get('Timestamp') <= end_date]

            order = query.get('order', [None])[0]
            if order == 'sentiment':
                filtered_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse= True)
                                          
            # Create the response body from the reviews and convert to a JSON 
            # byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        elif environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(content_length).decode('utf-8')   
                new_review_data = parse_qs(request_body)

                review_body = new_review_data.get('ReviewBody', [None])[0]
                location = new_review_data.get('Location', [None])[0]

                if not review_body or not location:
                    raise ValueError("Both 'ReviewBody' and 'Location' must be provided")
                
                if location not in ["San Diego, California", "Denver, Colorado"]:
                    raise ValueError('Invalid location')
                
                sentiment= self.analyze_sentiment(review_body)

                new_review={
                    "ReviewId" : str(uuid.uuid4()),
                    "Location" : location,
                    "Timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ReviewBody" : review_body,
                    "sentiment" : sentiment
        
                }

                reviews.append(new_review)

                response_body = json.dumps(new_review, indent=2).encode("utf-8")    

                start_response("201 Created", [ 
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            except Exception as e:
                error_response = json.dumps({"error": str(e)}).encode("utf-8")
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(error_response)))
                ])
                return [error_response] 


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()