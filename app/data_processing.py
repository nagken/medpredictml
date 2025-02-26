import pandas as pd

def load_and_process_data():
    """Load dataset, clean, and prepare for ML model training."""
    file_path = "C:/projects/MedPredictML/data/diabetic_data.csv"
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df.drop(columns=["encounter_id", "patient_nbr"], inplace=True)

    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Handle missing values
    df.fillna(0, inplace=True)

    print("✅ Data processed successfully!")
    return df
