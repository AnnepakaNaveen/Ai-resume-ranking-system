import streamlit as st
from PyPDF2 import PdfReader
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def extract_text(file):
    text = ""

    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            except:
                pass

    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text

    return text.lower()
def process_job_description(jd):
    jd = jd.lower()
    
    skills = re.split(r',|\n|\s{2,}', jd)

    if len(skills) == 1:
        skills = jd.split()

    skills = [s.strip() for s in skills if s.strip() != ""]
    
    return skills
def skill_match(resume, jd_list):
    count = 0
    for skill in jd_list:
        if skill in resume:
            count += 1
    return count / len(jd_list) if len(jd_list) > 0 else 0
def jaccard_similarity(resume, jd_list):
    resume_set = set(resume.split())
    jd_set = set(" ".join(jd_list).split())
    
    return len(resume_set & jd_set) / len(resume_set | jd_set)
st.title("📄 AI Resume Ranking System")

job_desc_input = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True
)
if st.button("Rank Resumes"):

    job_desc_list = process_job_description(job_desc_input)
    job_desc = " ".join(job_desc_list)

    resumes = []
    file_names = []

    for file in uploaded_files:
        text = extract_text(file)
        resumes.append(text)
        file_names.append(file.name)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_desc] + resumes)

    cosine_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    results = []

    for i, resume in enumerate(resumes):
        jaccard = jaccard_similarity(resume, job_desc_list)
        skill = skill_match(resume, job_desc_list)

        final_score = (
            0.5 * cosine_scores[i] +
            0.3 * jaccard +
            0.2 * skill
        )

        results.append({
            "name": file_names[i],
            "cosine": cosine_scores[i],
            "jaccard": jaccard,
            "skill": skill,
            "final": final_score
        })

    # Sort
    results = sorted(results, key=lambda x: x["final"], reverse=True)

    st.subheader("Ranked Resumes")

    for i, res in enumerate(results, 1):
        st.write(
            f"{i}. {res['name']} → "
            f"Cosine: {res['cosine']:.2f}%, "
            f"Jaccard: {res['jaccard']:.2f}%, "
            f"Skill: {res['skill']:.2f}%, "
            f"Final: {res['final']*100:.2f}%"
        )
    results = sorted(results, key=lambda x: x["final"], reverse=True)
    # 🎯 Recommended Resume (Top Match)
    if len(results) > 0:
         best = results[0]

    st.success("✅ Recommended Resume for this Job Description")

    st.write(
        f"📌 {best['name']} → "
        f"Final Score: {best['final']*100:.2f}%"
    )
    
