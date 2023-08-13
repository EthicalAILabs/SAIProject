import pandas as pd

def normalize_data(data_path):
    data = pd.read_csv(data_path)
    
    # Dummy normalization
    data['reported_crime'] = data['reported_crime'].str.lower()
    
    # Hidden comment with a riddle
    # "The key to the next part lies not in data, but in the flow of data. Follow the tokens."
    
    return data
