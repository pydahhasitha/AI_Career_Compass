# AI_Career_Compass
Student Placement Prediction & Career Guidance System

Overview

AI Career Compass is an AI-powered career guidance platform that helps students evaluate their placement readiness using Machine Learning and Generative AI.

The system predicts placement probability, analyzes student profiles, identifies skill gaps, provides certification recommendations, generates personalized learning roadmaps, and analyzes resumes using Google Gemini API.

Features
Placement Prediction
Predicts placement probability
Predicts placement status
Uses student academic and skill-related information
Career Guidance
AI-generated career recommendations
Skill gap analysis
Certification suggestions
Personalized learning roadmap
Resume Analyzer
Upload PDF resumes
Resume scoring
Missing skill identification
Resume improvement suggestions
Dashboard
Placement probability visualization
Career insights
Resume analysis results
Student profile overview
Dataset

Dataset Used:

student_placement_career_success_dataset_2026.csv

The dataset contains student-related information such as:

CGPA
Projects
Internships
Certifications
Communication Skills
Coding Skills
Aptitude Scores
Placement Status
Technology Stack
Frontend
Streamlit
Backend
Python
Machine Learning
Scikit-Learn
Random Forest Classifier
Data Processing
Pandas
NumPy
Generative AI
Google Gemini API
Resume Processing
PDFPlumber
Model Storage
Joblib
Development Environment
Google Colab
Deployment
Streamlit Community Cloud

## Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Train-Test Split
   │
   ▼
Random Forest Model
   │
   ▼
Model Evaluation
   │
   ▼
Model Saving
   │
   ▼
Streamlit Frontend
   │
   ▼
Gemini Integration
   │
   ▼
Resume Analyzer
   │
   ▼
Deployment
```
Machine Learning Model
Algorithm Used

Random Forest Classifier

Why Random Forest?
High Accuracy
Handles Large Datasets
Reduces Overfitting
Works Well for Classification Problems
Supports Probability Predictions
Installation
Clone Repository
git clone https://github.com/yourusername/AI_Career_Compass.git

cd AI_Career_Compass
Create Virtual Environment
python -m venv venv
Activate Environment

Windows:

venv\Scripts\activate

Linux / Mac:

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Environment Variables

Create a .streamlit/secrets.toml file:

GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

Or create a .env file:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
Running the Application
streamlit run app.py
## Running the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Project Structure

```text
AI_Career_Compass/
│
├── app.py
├── train_model.py
├── placement_model.pkl
├── requirements.txt
├── README.md
│
├── data/
│   └── student_placement_career_success_dataset_2026.csv
│
├── pages/
│   ├── Placement_Predictor.py
│   ├── Career_Guide.py
│   └── Resume_Analyzer.py
│
├── models/
│   └── placement_model.pkl
│
└── assets/
```
Placement Prediction

Input:

CGPA
Projects
Internships
Certifications

Output:

Placement Probability: 87%
Status: Likely to be Placed
Career Guidance
Recommended Career Paths:
- Data Analyst
- Data Scientist
- ML Engineer
Resume Analysis
Resume Score: 82/100

Missing Skills:
- SQL
- Power BI
- Machine Learning
Future Scope
LinkedIn Integration
Job Recommendation Engine
Internship Recommendation System
ATS Resume Scoring
Interview Preparation Module
Mock Interview Chatbot
Mobile Application
Deep Learning Models
Multi-language Support
Learning Outcomes

This project demonstrates:

Machine Learning
Data Preprocessing
Classification Models
Feature Engineering
Streamlit Development
Generative AI Integration
Resume Parsing
Deployment
End-to-End AI Application Development
Author

Pydah Hasitha Sai Keerthana

B.Tech CSE (CS&IT)
KL Deemed to be University
