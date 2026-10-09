import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load the job dataset
data = pd.read_csv("data/jobs.csv")

# Combine job details into one text field
data["Job_Info"] = (
    data["Application"] + " " +
    data["Job_Type"] + " " +
    data["Frequency"] + " " +
    data["Platform"] + " " +
    data["Dependency"]
)

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["Job_Info"])
# Calculate similarity between all jobs
similarity = cosine_similarity(tfidf_matrix)

print("\nSimilarity matrix created successfully!")
print("Similarity matrix shape:", similarity.shape)
# Display results
print("TF-IDF model created successfully!")
print("Number of jobs:", len(data))
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Function to recommend similar jobs
def recommend_jobs(job_name, top_n=5):
    job_index = data[data["Job_Name"] == job_name].index[0]

    similarity_scores = list(enumerate(similarity[job_index]))
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )
    recommendations = []
    for index, score in similarity_scores[1:top_n + 1]:
        recommendations.append(
            (data.iloc[index]["Job_Name"], score)
        )
        return recommendations
# Test the recommendation system
print("\nRecommendations for PAYROLL_VALIDATION:")
print(recommend_jobs("PAYROLL_VALIDATION"))
print("\nRecommendations for SALES_VALIDATION:")
print(recommend_jobs("SALES_VALIDATION"))
print("\nRecommendations for INVOICE_VALIDATION:")
print(recommend_jobs("INVOICE_VALIDATION"))

# Save the TF-IDF model
joblib.dump(vectorizer, "Model/tfidf_vectorizer.pkl")