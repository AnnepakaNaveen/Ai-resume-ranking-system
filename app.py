import base64
import streamlit as st

def set_bg():
    with open("download.jpg", "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(f"""
    <style>

    /* DARK OVERLAY (MAIN FIX) */
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
            url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}

    /* TITLE */
    h1 {{
        color: #ffffff !important;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }}

    /* LABEL TEXT */
    label {{
        color: #f1f5f9 !important;
        font-size: 16px;
    }}

    /* TEXT AREA */
    textarea {{
        background-color: rgba(255,255,255,0.95) !important;
        color: black !important;
        border-radius: 10px !important;
    }}

    /* UPLOAD BOX */
    .stFileUploader {{
        background-color: rgba(255,255,255,0.2);
        padding: 10px;
        border-radius: 10px;
    }}

    /* BUTTON */
    .stButton button {{
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 8px 20px;
    }}

    </style>
    """, unsafe_allow_html=True)

set_bg()
import streamlit as st
import numpy as np
import re
import cv2
import pytesseract
from pypdf import PdfReader
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

def set_bg():
    with open("download.jpg", "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()   

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()

def extract_text_fast(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages[:5]:
            text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text
    return clean_text(text)

def extract_text_from_image(file):
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    text = pytesseract.image_to_string(thresh)
    return clean_text(text)

def skill_match(resume, jd_list):
    return sum(1 for skill in jd_list if skill in resume) / max(len(jd_list), 1)

def jaccard_similarity(resume, jd_list):
    resume_set = set(resume.split())
    jd_set = set(" ".join(jd_list).split())
    return len(resume_set & jd_set) / max(len(resume_set | jd_set), 1)

st.title("📄 Resume Ranking System")

job_desc_input = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.button("Rank Resumes"):

    resumes = []
    file_names = []

    jd_list = re.split(r',|\n|\s+', job_desc_input.lower())
    jd_list = [s for s in jd_list if s]
    job_desc = " ".join(jd_list)

    for file in uploaded_files:
        if file.name.endswith(("png", "jpg", "jpeg")):
            text = extract_text_from_image(file)
        else:
            text = extract_text_fast(file)

        if len(text.strip()) > 50:
            resumes.append(text)
            file_names.append(file.name)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_desc] + resumes)

    cosine_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    results = []

    for i, resume in enumerate(resumes):
        cos = cosine_scores[i]
        jac = jaccard_similarity(resume, jd_list)
        skill = skill_match(resume, jd_list)

        final = (0.7 * cos) + (0.2 * jac) + (0.1 * skill)

        results.append((file_names[i], final * 100))

    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader(" Results")
    for name, score in results:
        st.write(f"{name} → {score:.2f}%")
