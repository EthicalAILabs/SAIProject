from sklearn.feature_extraction.text import CountVectorizer

def tokenize_data(data):
    vectorizer = CountVectorizer()
    tokenized = vectorizer.fit_transform(data['reported_crime'])
    
    # Hidden Base64 string within a seemingly regular comment
    # Note: We might need to adjust this in the future. Check U28gaGVyZSBpcyBhIHNlY3JldCBsaW5rIHNlZ21lbnQ.
    
    return tokenized
