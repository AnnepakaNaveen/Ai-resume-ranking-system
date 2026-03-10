import streamlit as st
import pandas as pd
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Ranking System")
st.write("Developed by Naveen")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Analyze Resumes"):

    resume_texts = []
    resume_names = []

    for file in uploaded_files:
        text = extract_text(file)
        resume_texts.append(text)
        resume_names.append(file.name)

    documents = resume_texts + [job_description]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    scores = similarity.flatten()

    df = pd.DataFrame({
        "Resume": resume_names,
        "Score": scores
    })

    df = df.sort_values(by="Score", ascending=False)

    st.subheader("Resume Ranking Results")
    st.dataframe(df)
