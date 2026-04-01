# Resume Intelligence Pro 🚀

An advanced AI-powered resume screening system that predicts recruiter decisions with high precision using Machine Learning.

## 🏗️ Project Architecture

```text
mobile-classification-project/
│
├── backend/
│   ├── app.py              # Flask API (entry point)
│   ├── model.py            # ML logic (training & prediction)
│   ├── model.pkl           # Saved model state
│   └── requirements.txt    # Dependencies
│
├── frontend/
│   ├── index.html          # Premium UI
│   ├── style.css           # Glassmorphic Styling
│   └── script.js           # API Communication
│
├── dataset/
│   └── AI_Resume_Screening.csv  # HR Dataset
│
└── notebook/
    └── model_training.ipynb # Data Science Playground
```

## ⚡ Features

- **Premium UI**: Modern dark mode experience with glassmorphism and fluid animations.
- **AI-Powered**: Uses a Random Forest Classifier trained on real-world hiring patterns.
- **Robust Preprocessing**: Handles multi-label skills, categorical education/roles, and feature scaling.
- **Instant Results**: Real-time prediction of "Hire" or "Reject" status.

## 🛠️ Setup & Installation

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The API will start at `http://localhost:5000`.

### 2. Frontend Setup
Simply open `frontend/index.html` in any modern web browser.

## 🧠 Model Details
The system uses a **Random Forest Classifier** which performs exceptionally well on tabular HR data. It processes:
- **Numerical Features**: Experience, Salary Expectations, Projects, AI Score.
- **Categorical Features**: Education, Job Role (Label Encoded).
- **Multi-Label Features**: Skills (Binarized).
- **Scaling**: All numeric features are standardized using `StandardScaler`.

## 🚀 Live Demo
[Click here to view the app] => (https://resume-screening-ai-gd9p.vercel.app/))
