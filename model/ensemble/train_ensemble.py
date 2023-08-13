from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd

data = pd.read_csv('../../data/processed_data/tokenized_data.csv')
X = data.drop(columns=['prediction'])
y = data['prediction']

# Placeholder ensemble training
rf_model = RandomForestClassifier()
gboost_model = GradientBoostingClassifier()

rf_model.fit(X, y)
gboost_model.fit(X, y)

# Hidden message in a lambda function (seems benign but isn't)
link_extractor = lambda x: x[:4] + '-' + x[5:9] if len(x) == 9 else None
secret_link_part = link_extractor("link-1234")
