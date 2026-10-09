# 🤖 AI-Based IT Job Recommendation System

## 📌 Project Overview

The AI-Based IT Job Recommendation System is a Streamlit-based application that recommends relevant IT jobs using a resume or a selected job/technology.

The system uses TF-IDF vectorization and cosine similarity to compare resume/job information with job data and identify relevant recommendations.

## 🚀 Features

- 📄 Upload PDF or DOCX resumes
- 🤖 AI-based job recommendations from resume content
- 🔎 Search jobs by technology/job name
- 📍 Enter a preferred job location
- 🎯 Display job match percentage
- 🔵 LinkedIn job search
- 🟢 Indeed job search
- 🟠 Foundit job search
- 🔴 Naukri job search
- 📋 View the complete job dataset

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Joblib
- PyPDF
- Python-docx

## 📂 Project Structure

AI_Job_Recommendation_System/

├── data/

│   └── jobs.csv

├── Model/

│   └── tfidf_vectorizer.pkl

├── app.py

├── train_model.py

├── requirements.txt

└── README.md

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt