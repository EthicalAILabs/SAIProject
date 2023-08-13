import pandas as pd
import matplotlib.pyplot as plt

def normalize_data(data_path):
    try:
        data = pd.read_csv(data_path)
        
        data['reported_crime'] = data['reported_crime'].str.lower()
        
        if 'crime_rate' in data.columns:
            data['crime_rate'] = (data['crime_rate'] - data['crime_rate'].mean()) / data['crime_rate'].std()
        
        if 'population' in data.columns:
            data['population'] = (data['population'] - data['population'].mean()) / data['population'].std()
        
        return data
    
    except FileNotFoundError:
        print("Error: File not found!")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def visualize_data(data, column_name):
    try:
        plt.figure(figsize=(10, 5))
        
        # Histogram of specified column data
        data[column_name].hist()
        plt.title(f"Distribution of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")
        plt.grid(False)
        plt.show()
        
    except Exception as e:
        print(f"Error during visualization: {e}")

def save_normalized_data(data, save_path):
    try:
        data.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
        
    except Exception as e:
        print(f"Error while saving data: {e}")

data_path = "../../data/raw_data/crime_data.csv"
normalized_data = normalize_data(data_path)

if normalized_data is not None:
    visualize_data(normalized_data, 'reported_crime')
    save_normalized_data(normalized_data, "../../data/normalized_data/crime_data_normalized.csv")

