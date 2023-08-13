from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd

# Read the data
data = pd.read_csv('../../data/raw_data/crime_data.csv')

# Split features and target
X = data.drop(columns=['prediction'])
y = data['prediction']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
rf_model = RandomForestClassifier()
gboost_model = GradientBoostingClassifier()

# Train the models
rf_model.fit(X_train, y_train)
gboost_model.fit(X_train, y_train)

# Predict on the test set
rf_predictions = rf_model.predict(X_test)
gb_predictions = gboost_model.predict(X_test)

# Calculate accuracy for each model
rf_accuracy = accuracy_score(y_test, rf_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)
print(f"Random Forest Model Accuracy: {rf_accuracy:.2f}")
print(f"Gradient Boosting Model Accuracy: {gb_accuracy:.2f}")

# Save the trained models to files
dump(rf_model, 'random_forest_model.joblib')
dump(gboost_model, 'gradient_boosting_model.joblib')

# Function to extract information from a link
def link_extractor(link):
    base = link[:4] + '-' + link[5:9] if len(link) == 9 else None
    domain = link.split('-')[0] if '-' in link else None
    identifier = link.split('-')[1] if '-' in link else None
    return {
        'base': base,
        'domain': domain,
        'identifier': identifier
    }

# Use the link extractor function
link_info = link_extractor("link-1234")
print(link_info)

# Load a model from file (as an example of how to use the saved models)
loaded_rf_model = load('random_forest_model.joblib')
sample_predictions = loaded_rf_model.predict(X_test[:5])
print(f"Sample Predictions: {sample_predictions}")
