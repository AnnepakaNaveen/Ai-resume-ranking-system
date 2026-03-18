import streamlit as st
from PyPDF2 import PdfReader
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()

def extract_text(file):

    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except:
                pass
        return clean_text(text)

    elif file.name.endswith(".txt"):
        return clean_text(file.read().decode("utf-8"))

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return clean_text(" ".join([para.text for para in doc.paragraphs]))

    return ""

def rank_resumes(job_description, resumes):

    job_description = clean_text(job_description)

    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_vec = vectors[0]
    resume_vecs = vectors[1:]

    cosine_scores = cosine_similarity([job_vec], resume_vecs).flatten()

    results = []

    for i, resume in enumerate(resumes):

        set1 = set(job_description.split())
        set2 = set(resume.split())

        jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
        skill = len(set1 & set2) / len(set1) if len(set1) > 0 else 0

        final_score = (0.5 * cosine_scores[i]) + (0.3 * jaccard) + (0.2 * skill)

        results.append({
            "cosine": cosine_scores[i],
            "jaccard": jaccard,
            "skill": skill,
            "final": final_score
        })

    return results


st.title("AI Resume Screening & Ranking System")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Rank Resumes"):

    if not job_description:
        st.warning("Enter job description")

    elif not uploaded_files:
        st.warning("Upload resumes")

    else:
        resumes = [extract_text(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)

        ranked_resumes = sorted(
            zip(uploaded_files, scores),
            key=lambda x: x[1]['final'],
            reverse=True
        )

        st.subheader("Detailed Resume Ranking")

        st.subheader("Ranked Resumes (All Scores)")

        for i, (file, score) in enumerate(ranked_resumes, start=1):

           st.write(
              f"{i}. {file.name} → "
              f"Cosine: {score['cosine']:.3f}, "
              f"Jaccard: {score['jaccard']:.3f}, "
              f"Skill: {score['skill']:.3f}, "
              f"Final: {score['final']*100:.2f}"
          )
