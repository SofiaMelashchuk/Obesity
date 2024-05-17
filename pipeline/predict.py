import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Define paths to data and model directories
project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')
model_dir = os.path.join(project_dir, 'models')

# Path to the input CSV file for prediction
csv_file_path = os.path.join(data_dir, 'new_data_for_prediction.csv')

# Check if the CSV file exists
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"File '{csv_file_path}' not found. Please verify the file path.")

# Step 1: Reading New Data for Prediction
new_data = pd.read_csv(csv_file_path)
if new_data.empty:
    raise ValueError("Input data is empty. Please provide valid data for prediction.")

# Step 2: Data Processing (Applying Same Transformations as Training Data)
def preprocess_data(input_data):
    # Encode categorical features
    label_encoder = LabelEncoder()
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    for col in categorical_cols:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    return input_data

# Apply preprocessing to the new data
new_data_processed = preprocess_data(new_data)

# Load the trained model
model_path = os.path.join(model_dir, 'trained_model.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model '{model_path}' not found. Please train a model first.")

# Load the trained RandomForestClassifier model
model = joblib.load(model_path)

# Step 3: Prepare Input for Prediction
features_used = ['Age','Gender','Height','Weight','CALC','FAVC','FCVC','NCP',
                 'SCC','SMOKE','CH2O','family_history_with_overweight','FAF',
                 'TUE','CAEC','MTRANS']

# Ensure all required features are present in the input data
missing_features = set(features_used) - set(new_data_processed.columns)
if missing_features:
    raise ValueError(f"Missing required features in input data: {missing_features}")

X_pred = new_data_processed[features_used]

# Step 4: Scaling Features (if necessary)
scaler = StandardScaler()
X_pred_scaled = scaler.fit_transform(X_pred)  # Apply the same scaler used for training

# Step 5: Predicting the Result
predictions = model.predict(X_pred_scaled)

# Step 6: Formatting and Saving the Prediction Result
prediction_result = pd.DataFrame({'Prediction': predictions})

# Add original data columns for reference (optional)
prediction_result = pd.concat([new_data.reset_index(drop=True), prediction_result], axis=1)

output_file_path = os.path.join(data_dir, 'prediction_result.csv')
prediction_result.to_csv(output_file_path, index=False)

print(f"Prediction completed. Results saved to '{output_file_path}'.")
