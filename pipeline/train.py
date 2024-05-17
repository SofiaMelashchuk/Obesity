import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Define paths to data and output directories
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

# Path to the input CSV file
csv_file_path = os.path.join(data_dir, 'obesity.csv')

# Check if the CSV file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"File '{csv_file_path}' not found. Please verify the file path.")

# Step 1: Dataset Reading
data = pd.read_csv(csv_file_path)

# Step 2: Data Processing
data.dropna(inplace=True)

# Encode categorical labels
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['family_history_with_overweight'] = label_encoder.fit_transform(data['family_history_with_overweight'])
data['FAVC'] = label_encoder.fit_transform(data['FAVC'])
data['CAEC'] = label_encoder.fit_transform(data['CAEC'])
data['SMOKE'] = label_encoder.fit_transform(data['SMOKE'])
data['SCC'] = label_encoder.fit_transform(data['SCC'])
data['CALC'] = label_encoder.fit_transform(data['CALC'])
data['MTRANS'] = label_encoder.fit_transform(data['MTRANS'])

# Split features and target
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Step 3: Dataset Division
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Validation of Results
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Validation Accuracy: {accuracy}')
validation_results_path = os.path.join(output_dir, 'validation_results.txt')
with open(validation_results_path, 'w') as f:
    f.write(f'Validation Accuracy: {accuracy}')

# Create models directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define paths for saving model and other outputs
model_save_path = os.path.join(output_dir, 'trained_model.pkl')
feature_importances_path = os.path.join(output_dir, 'feature_importances.csv')

# Step 7: Save Trained Model
joblib.dump(model, model_save_path)

# Step 8: Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df.to_csv(feature_importances_path, index=False)

# Print success message
print("Model training and data processing completed successfully!")