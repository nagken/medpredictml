import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
DATA_PATH = "C:/projects/MedPredictML/data/diabetic_data.csv"
df = pd.read_csv(DATA_PATH)

# Convert 'age' column from object to numeric range
age_mapping = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45, 
    '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
}
df['age'] = df['age'].map(age_mapping)

# Feature Engineering - Selecting relevant features
df = df[['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
         'num_medications', 'number_diagnoses', 'readmitted']]

df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == 'YES' else 0)

# Splitting Data
X = df.drop(columns=['readmitted'])
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained with accuracy: {accuracy:.4f}")

# Save the Model
MODEL_PATH = "models/xgboost_patient_readmission.pkl"
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
