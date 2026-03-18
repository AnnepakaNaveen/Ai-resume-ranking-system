import streamlit as st
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

    st.subheader("🏆 Ranked Resumes")

    for i, res in enumerate(results, 1):
        st.write(
            f"{i}. {res['name']} → "
            f"Cosine: {res['cosine']*100:.2f}%, "
            f"Jaccard: {res['jaccard']*100:.2f}%, "
            f"Skill: {res['skill']*100:.2f}%, "
            f"Final: {res['final']*100:.2f}%"
        )
