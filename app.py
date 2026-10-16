import streamlit as st
import re
import io
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Resume + Job Match Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #222222;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #666666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #222222;
    margin-top: 20px;
    margin-bottom: 12px;
}

.skill-item {
    font-size: 17px;
    font-weight: 600;
    color: #222222;
    margin: 7px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📄 Resume + Job Match Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze your resume against a job description using NLP'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SKILLS DATABASE
# =========================================================

skills_list = [
    "python",
    "java",
    "c++",
    "c",
    "sql",
    "mysql",
    "html",
    "css",
    "javascript",
    "react",
    "node.js",
    "flask",
    "django",
    "streamlit",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "pandas",
    "numpy",
    "scikit-learn",
    "git",
    "github",
    "docker",
    "aws",
    "communication",
    "problem solving",
    "teamwork",
    "leadership",
    "data analysis",
    "power bi",
    "excel"
]


# =========================================================
# RESUME TEXT EXTRACTION
# =========================================================

def extract_resume_text(uploaded_file):

    pdf_bytes = uploaded_file.read()

    pdf_file = io.BytesIO(pdf_bytes)

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + " "

    return text


# =========================================================
# SKILL EXTRACTION
# =========================================================

def find_skills(text):

    text = text.lower()

    found_skills = []

    for skill in skills_list:

        # Special handling for C
        if skill == "c":
            pattern = r"(?<![a-z])c(?![a-z+])"

        # Special handling for C++
        elif skill == "c++":
            pattern = r"(?<![a-z])c\+\+(?![a-z])"

        else:
            pattern = (
                r"(?<![a-z])"
                + re.escape(skill)
                + r"(?![a-z])"
            )

        if re.search(pattern, text):

            found_skills.append(skill)

    return found_skills


# =========================================================
# STEP 1 - UPLOAD RESUME
# =========================================================

st.markdown(
    '<div class="section-title">1️⃣ Upload Your Resume</div>',
    unsafe_allow_html=True
)

uploaded_resume = st.file_uploader(
    "Upload your resume",
    type=["pdf"]
)


# =========================================================
# STEP 2 - JOB DESCRIPTION
# =========================================================

st.markdown(
    '<div class="section-title">2️⃣ Enter Job Description</div>',
    unsafe_allow_html=True
)

job_description = st.text_area(
    "",
    height=220,
    placeholder="Paste the job description here..."
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Resume", use_container_width=True):

    # Check resume
    if uploaded_resume is None:

        st.warning(
            "Please upload your resume first."
        )

        st.stop()


    # Check job description
    if not job_description.strip():

        st.warning(
            "Please enter the job description."
        )

        st.stop()


    # =====================================================
    # EXTRACT RESUME
    # =====================================================

    resume_text = extract_resume_text(
        uploaded_resume
    )


    if not resume_text.strip():

        st.error(
            "Could not extract text from the resume."
        )

        st.stop()


    # =====================================================
    # CLEAN TEXT
    # =====================================================

    resume_clean = resume_text.lower()

    job_clean = job_description.lower()


    # =====================================================
    # CALCULATE NLP MATCH
    # =====================================================

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(
        [
            resume_clean,
            job_clean
        ]
    )

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0][0]

    match_percentage = round(
        similarity * 100,
        2
    )


    # =====================================================
    # FIND SKILLS
    # =====================================================

    resume_skills = find_skills(
        resume_clean
    )

    job_skills = find_skills(
        job_clean
    )


    # =====================================================
    # MATCHING SKILLS
    # =====================================================

    matching_skills = sorted(
        set(resume_skills)
        &
        set(job_skills)
    )


    # =====================================================
    # MISSING SKILLS
    # =====================================================

    missing_skills = sorted(
        set(job_skills)
        -
        set(resume_skills)
    )


    # =====================================================
    # ANALYSIS RESULT
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Analysis Result</div>',
        unsafe_allow_html=True
    )


    # Result metrics

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Resume Match",
            f"{match_percentage}%"
        )


    with col2:

        st.metric(
            "Matching Skills",
            len(matching_skills)
        )


    with col3:

        st.metric(
            "Missing Skills",
            len(missing_skills)
        )


    # =====================================================
    # MATCHING SKILLS
    # =====================================================

    st.markdown(
        '<div class="section-title">✅ Matching Skills</div>',
        unsafe_allow_html=True
    )


    if matching_skills:

        for skill in matching_skills:

            st.markdown(
                f'<div class="skill-item">• {skill.lower()}</div>',
                unsafe_allow_html=True
            )

    else:

        st.write(
            "No matching skills detected."
        )


    # =====================================================
    # MISSING SKILLS
    # =====================================================

    st.markdown(
        '<div class="section-title">❌ Missing Skills</div>',
        unsafe_allow_html=True
    )


    if missing_skills:

        for skill in missing_skills:

            st.markdown(
                f'<div class="skill-item">• {skill.lower()}</div>',
                unsafe_allow_html=True
            )

    else:

        st.write(
            "No missing skills detected."
        )


    # =====================================================
    # SUGGESTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Suggestions</div>',
        unsafe_allow_html=True
    )


    if missing_skills:

        st.write(
            "Consider learning or highlighting "
            "these skills in your resume:"
        )


        for skill in missing_skills:

            st.markdown(
                f'<div class="skill-item">• {skill.lower()}</div>',
                unsafe_allow_html=True
            )

    else:

        st.success(
            "Your resume contains the main skills "
            "mentioned in the job description."
        )


    # =====================================================
    # EXTRACTED RESUME TEXT
    # =====================================================

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.write(
            resume_text
        )