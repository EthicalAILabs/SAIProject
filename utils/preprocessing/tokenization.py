import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def tokenize_data(data, use_ngrams=False):
    """
    Tokenizes the 'reported_crime' column in the provided data.
    
    Args:
    - data (DataFrame): Input data with a 'reported_crime' column.
    - use_ngrams (bool): If True, use bi-gram tokenization.

    Returns:
    - DataFrame: A DataFrame with tokenized features.
    """
    # Handle potential missing values
    data['reported_crime'] = data['reported_crime'].fillna("")
    
    ngram_range = (1, 2) if use_ngrams else (1, 1)
    
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    tokenized = vectorizer.fit_transform(data['reported_crime'])
    
    # Convert sparse matrix to DataFrame for easier handling
    tokenized_df = pd.DataFrame(tokenized.toarray(), columns=vectorizer.get_feature_names_out())
    
    return tokenized_df

def visualize_most_frequent_tokens(tokenized_data, top_n=10):
    """
    Visualizes the most frequent tokens in the tokenized data.
    
    Args:
    - tokenized_data (DataFrame): Data with tokenized features.
    - top_n (int): Number of top tokens to display.
    """
    sum_words = tokenized_data.sum(axis=0)
    top_words = sum_words.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 5))
    top_words.plot(kind='barh')
    plt.title('Most Frequent Tokens')
    plt.gca().invert_yaxis()  # Highest values at the top
    plt.show()

def display_vocabulary(vectorizer):
    """
    Displays the vocabulary learned by the vectorizer.
    
    Args:
    - vectorizer (CountVectorizer): Fitted vectorizer object.
    """
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary Learned:")
    print(", ".join(vocabulary))

# Example usage
data = pd.DataFrame({
    'reported_crime': ["Theft in the park", "Burglary at downtown store", "Robbery on 5th street", "Vandalism at school"]
})

tokenized_data = tokenize_data(data, use_ngrams=True)

visualize_most_frequent_tokens(tokenized_data)
