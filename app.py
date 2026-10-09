import streamlit as st
import pandas as pd
import joblib
from pypdf import PdfReader
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus

# -----------------------------
# Load data and model
# -----------------------------

data = pd.read_csv("data/jobs.csv")
vectorizer = joblib.load("Model/tfidf_vectorizer.pkl")

# Create job information
data["Job_Info"] = (
    data["Job_Name"].fillna("") + " " +
    data["Application"].fillna("") + " " +
    data["Job_Type"].fillna("") + " " +
    data["Frequency"].fillna("") + " " +
    data["Platform"].fillna("") + " " +
    data["Dependency"].fillna("")
)

# TF-IDF
tfidf_matrix = vectorizer.transform(data["Job_Info"])

# Similarity
similarity = cosine_similarity(tfidf_matrix)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="AI IT Job Recommendation System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Based IT Job Recommendation System")

st.write(
    "Find similar IT jobs and quickly search for real job opportunities "
    "across major job portals."
)

st.divider()


# -----------------------------
# Resume Upload
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Upload Your Resume")


def extract_resume_text(resume_file):

    text = ""

    if resume_file.name.lower().endswith(".pdf"):

        reader = PdfReader(resume_file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "


    elif resume_file.name.lower().endswith(".docx"):

        document = Document(resume_file)

        for paragraph in document.paragraphs:
            text += paragraph.text + " "


    return text


resume = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"]
)


if resume:

    resume_text = extract_resume_text(resume)

    if resume_text.strip():

        st.success(f"Resume uploaded successfully: {resume.name}")

        st.info(
            f"Resume text extracted successfully. "
            f"Characters detected: {len(resume_text)}"
        )

        st.text_area(
    "📋 Extracted Resume Text",
    resume_text,
    height=250
)

    else:

        st.warning(
            "Could not extract text from this resume."
        )


st.divider()

with col2:

    st.subheader("🔎 Search Job & Location")

search_job = st.selectbox(
    "Job / Technology",
    sorted(data["Job_Name"].dropna().unique())
)

search_location = st.text_input(
        "Location",
        placeholder="Example: Hyderabad, India"
    )

search_button = st.button(
        "🔎 Search Jobs",
        use_container_width=True
    )

if search_button:

        if search_job.strip():

            search_text = search_job.strip()

            search_vector = vectorizer.transform([search_text])

            search_similarity = cosine_similarity(
                search_vector,
                tfidf_matrix
            )[0]

            top_indices = search_similarity.argsort()[::-1][:10]

            st.session_state["search_results"] = top_indices

            st.session_state["search_similarity"] = search_similarity

            st.session_state["search_location"] = search_location

        else:

            st.warning("Please enter a Job / Technology to search.")

# -----------------------------
# Search Results
# -----------------------------

if "search_results" in st.session_state:

    st.subheader("🎯 Recommended Jobs")

    for index in st.session_state["search_results"]:

        st.markdown(
            f"### 💼 {data.iloc[index]['Job_Name']}"
        )

        st.write(
            f"**Application:** {data.iloc[index]['Application']}"
        )

        st.write(
            f"**Job Type:** {data.iloc[index]['Job_Type']}"
        )

        st.write(
            f"**Platform:** {data.iloc[index]['Platform']}"
        )

        st.write(
            f"**Match:** {st.session_state['search_similarity'][index] * 100:.2f}%"
        )

        search_text = (
            f"{data.iloc[index]['Job_Name']} "
            f"{data.iloc[index]['Application']} "
            f"{data.iloc[index]['Job_Type']}"
        )

        encoded_job = quote_plus(search_text)

        location = st.session_state.get(
            "search_location",
            "India"
        )

    location = st.session_state.get(
    "search_location",
    "India"
    )
    portal1, portal2, portal3, portal4 = st.columns(4)

    with portal1:
            st.link_button(
                "🔵 LinkedIn",
                f"https://www.linkedin.com/jobs/search/?keywords={encoded_job}&location={quote_plus(location)}",
                use_container_width=True
            )

    with portal2:
            st.link_button(
                "🟢 Indeed",
                f"https://in.indeed.com/jobs?q={encoded_job}&l={quote_plus(location)}",
                use_container_width=True
            )

    with portal3:
            st.link_button(
                "🟠 Foundit",
                f"https://www.foundit.in/search/jobs?query={encoded_job}&locations={quote_plus(location)}",
                use_container_width=True
            )

    with portal4:
            st.link_button(
                "🔴 Naukri",
                f"https://www.naukri.com/{encoded_job.replace('+', '-')}-jobs",
                use_container_width=True
            )

    st.divider()

# -----------------------------
# Resume-Based Recommendations
# -----------------------------

if resume and resume_text.strip():

    st.subheader("🤖 AI Job Recommendations Based on Resume")

    resume_vector = vectorizer.transform([resume_text])

    resume_similarity = cosine_similarity(
        resume_vector,
        tfidf_matrix
    )[0]

    top_indices = resume_similarity.argsort()[::-1][:50]

    st.write("Recommended jobs based on your resume:")

    for index in top_indices:

        st.markdown(
            f"### 💼 {data.iloc[index]['Job_Name']}"
        )

        search_text = (
            f"{data.iloc[index]['Job_Name']} "
            f"{data.iloc[index]['Application']} "
            f"{data.iloc[index]['Job_Type']}"
        )

        encoded_job = quote_plus(search_text)

        st.link_button(
            "🔵 Search LinkedIn Jobs",
            f"https://www.linkedin.com/jobs/search/?keywords={encoded_job}&location=India"
        )

        st.link_button(
            "🟢 Search Indeed Jobs",
            f"https://in.indeed.com/jobs?q={encoded_job}&l=India"
        )

        st.link_button(
            "🟠 Search Foundit Jobs",
            f"https://www.foundit.in/search/jobs?query={encoded_job}&locations=India"
        )

        st.link_button(
            "🔴 Search Naukri Jobs",
            f"https://www.naukri.com/{encoded_job.replace('+', '-')}-jobs"
        )

# -----------------------------
# All Jobs
# -----------------------------

st.subheader("📋 All Jobs in Dataset")

st.write(
    f"Currently available: **{len(data)} jobs** "
    f"across **{data['Application'].nunique()} applications/technologies**."
)

st.dataframe(
    data[
        [
            "Job_Name",
            "Application",
            "Job_Type",
            "Frequency",
            "Platform",
            "Dependency"
        ]
    ],
    use_container_width=True,
    height=500
)