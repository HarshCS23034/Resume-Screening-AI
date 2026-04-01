import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
import joblib
import os

# Paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'AI_Resume_Screening.csv')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def train_model():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Preprocessing Skills (MultiLabelBinarizer)
    df['Skills_List'] = df['Skills'].apply(lambda x: [s.strip() for s in x.split(',')] if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df['Skills_List'])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

    # Label Encoder for categorical columns
    le_education = LabelEncoder()
    df['Education_Encoded'] = le_education.fit_transform(df['Education'])

    le_job_role = LabelEncoder()
    df['Job_Role_Encoded'] = le_job_role.fit_transform(df['Job Role'])

    # Target variable
    le_decision = LabelEncoder()
    df['Decision_Encoded'] = le_decision.fit_transform(df['Recruiter Decision'])

    # Features
    numeric_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count', 'AI Score (0-100)', 'Education_Encoded', 'Job_Role_Encoded']
    X_numeric = df[numeric_cols]
    
    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled_df = pd.DataFrame(X_numeric_scaled, columns=numeric_cols)

    X = pd.concat([X_numeric_scaled_df, skills_df], axis=1)
    y = df['Decision_Encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and preprocessing tools
    model_data = {
        'model': model,
        'mlb': mlb,
        'scaler': scaler,
        'le_education': le_education,
        'le_job_role': le_job_role,
        'le_decision': le_decision,
        'features_order': X.columns.tolist(),
        'numeric_cols': numeric_cols
    }
    
    joblib.dump(model_data, MODEL_SAVE_PATH)
    print(f"Model trained and saved to {MODEL_SAVE_PATH}")
    print(f"Accuracy: {model.score(X_test, y_test):.2f}")

def predict_decision(input_data):
    """
    input_data: dict with keys matching the dataset
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        train_model()

    model_data = joblib.load(MODEL_SAVE_PATH)
    model = model_data['model']
    mlb = model_data['mlb']
    scaler = model_data['scaler']
    le_education = model_data['le_education']
    le_job_role = model_data['le_job_role']
    le_decision = model_data['le_decision']
    features_order = model_data['features_order']
    numeric_cols = model_data['numeric_cols']

    # Process new input skills
    new_skills = [s.strip() for s in input_data.get('Skills', '').split(',')]
    # Handle unseen skills by filtering them or using MLB's handling (mlb.transform handles them by ignoring)
    skills_encoded = mlb.transform([new_skills])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

    # Prepare numeric inputs
    new_input_numeric = pd.DataFrame([{
        'Experience (Years)': input_data.get('Experience (Years)', 0),
        'Salary Expectation ($)': input_data.get('Salary Expectation ($)', 0),
        'Projects Count': input_data.get('Projects Count', 0),
        'AI Score (0-100)': input_data.get('AI Score (0-100)', 0),
        'Education_Encoded': le_education.transform([input_data.get('Education', 'B.Sc')])[0],
        'Job_Role_Encoded': le_job_role.transform([input_data.get('Job Role', '')])[0],
    }])

    # Scale numeric input
    new_input_numeric_scaled = scaler.transform(new_input_numeric[numeric_cols])
    new_input_numeric_scaled_df = pd.DataFrame(new_input_numeric_scaled, columns=numeric_cols)

    # Combine with skills
    X_new = pd.concat([new_input_numeric_scaled_df, skills_df], axis=1)
    X_new = X_new[features_order] # Ensure same order

    prediction = model.predict(X_new)
    return le_decision.inverse_transform(prediction)[0]

if __name__ == "__main__":
    train_model()
