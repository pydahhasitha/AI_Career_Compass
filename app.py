import streamlit as st
import pandas as pd
import joblib
import json
import sklearn.compose._column_transformer as ct
import os
from google import genai
import pdfplumber
import re
# Compatibility fix for loading model saved with a different Scikit-Learn version
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass

    ct._RemainderColsList = _RemainderColsList

# Page configuration
st.set_page_config(
    page_title="AI Career Compass",
    page_icon="🎓",
    layout="wide"
)

# Load saved model and metadata
@st.cache_resource
def load_model():
    model = joblib.load("saved_model/placement_prediction_pipeline.pkl")

    # Compatibility fix for SimpleImputer across Scikit-Learn versions
    try:
        preprocessor = model.named_steps["preprocessor"]

        numeric_imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
        categorical_imputer = preprocessor.named_transformers_["cat"].named_steps["imputer"]

        if not hasattr(numeric_imputer, "_fill_dtype"):
            numeric_imputer._fill_dtype = "float64"

        if not hasattr(categorical_imputer, "_fill_dtype"):
            categorical_imputer._fill_dtype = object

    except Exception as e:
        st.warning(f"Model compatibility patch warning: {e}")

    with open("saved_model/model_metadata.json", "r") as file:
        metadata = json.load(file)

    return model, metadata
def get_gemini_response(prompt):
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)

        if not api_key:
            return generate_fallback_guidance()

        client = genai.Client(api_key=api_key)

        models_to_try = [
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ]

        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception:
                continue

        return generate_fallback_guidance()

    except Exception:
        return generate_fallback_guidance()
def generate_fallback_guidance():
    return """
    ### AI Guidance Temporarily Unavailable

    Gemini API quota is currently exhausted for this project, so the app is showing fallback career guidance.

    #### Career Direction
    Focus on roles that match your strongest profile areas such as software development, data analytics, machine learning, or cloud-based development.

    #### Strengths
    - Academic performance
    - Coding practice
    - Projects and certifications
    - Resume and interview preparation

    #### Skill Gap Analysis
    - Improve weak technical areas based on your dashboard scores.
    - Strengthen DSA, system design, and project explanation skills.
    - Build stronger GitHub and LinkedIn visibility.

    #### Certification Recommendations
    - Python for Data Science
    - Machine Learning Fundamentals
    - SQL and Database Management
    - Cloud Fundamentals
    - Resume and Interview Preparation

    #### Learning Roadmap
    1. Revise core programming and DSA.
    2. Build two strong portfolio projects.
    3. Improve resume with measurable project outcomes.
    4. Practice mock interviews weekly.
    5. Apply consistently to internships and entry-level roles.

    #### Final Action Plan
    - Update resume.
    - Improve GitHub profile.
    - Practice aptitude and communication.
    - Complete one relevant certification.
    - Build and deploy one end-to-end project.
    """
def extract_text_from_pdf(uploaded_file):
    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        return text.strip()

    except Exception as e:
        return f"PDF extraction error: {e}"


def identify_resume_skills(resume_text):
    skill_keywords = [
        "python", "java", "c++", "sql", "machine learning", "deep learning",
        "data science", "pandas", "numpy", "scikit-learn", "tensorflow",
        "pytorch", "nlp", "computer vision", "streamlit", "flask", "django",
        "html", "css", "javascript", "react", "node.js", "git", "github",
        "docker", "aws", "azure", "gcp", "excel", "power bi", "tableau",
        "data analysis", "statistics", "system design", "api", "mongodb",
        "mysql", "postgresql", "linux", "communication", "leadership"
    ]

    resume_lower = resume_text.lower()
    found_skills = []

    for skill in skill_keywords:
        if skill in resume_lower:
            found_skills.append(skill.title())

    return sorted(list(set(found_skills)))


def calculate_resume_score(resume_text, found_skills):
    resume_lower = resume_text.lower()
    words = resume_text.split()
    word_count = len(words)

    score = 0

    # Skill score: maximum 40
    skill_score = min(len(found_skills) * 4, 40)
    score += skill_score

    # Length score: maximum 15
    if 300 <= word_count <= 900:
        score += 15
    elif 150 <= word_count < 300 or 900 < word_count <= 1200:
        score += 10
    else:
        score += 5

    # Important sections score: maximum 25
    important_sections = {
        "education": ["education", "degree", "university", "college"],
        "skills": ["skills", "technical skills"],
        "projects": ["projects", "project"],
        "experience": ["experience", "internship", "work experience"],
        "certifications": ["certifications", "certificate", "courses"]
    }

    for section_keywords in important_sections.values():
        if any(keyword in resume_lower for keyword in section_keywords):
            score += 5

    # Contact/profile score: maximum 20
    if re.search(r"\S+@\S+", resume_text):
        score += 5

    if re.search(r"\b\d{10}\b", resume_text):
        score += 5

    if "linkedin" in resume_lower:
        score += 5

    if "github" in resume_lower:
        score += 5

    return min(score, 100)


def generate_basic_resume_feedback(resume_score, found_skills, resume_text):
    feedback = []

    resume_lower = resume_text.lower()

    if resume_score >= 80:
        feedback.append("Your resume looks strong and placement-ready.")
    elif resume_score >= 60:
        feedback.append("Your resume is decent, but it can be improved for better recruiter impact.")
    else:
        feedback.append("Your resume needs improvement before applying to competitive roles.")

    if len(found_skills) < 6:
        feedback.append("Add more relevant technical skills such as Python, SQL, Machine Learning, Git, and project-specific tools.")

    if "project" not in resume_lower:
        feedback.append("Add a dedicated Projects section with problem statement, tools used, and measurable outcomes.")

    if "internship" not in resume_lower and "experience" not in resume_lower:
        feedback.append("Add internship, training, freelance, or academic experience if available.")

    if "github" not in resume_lower:
        feedback.append("Add your GitHub profile link to showcase coding and project work.")

    if "linkedin" not in resume_lower:
        feedback.append("Add your LinkedIn profile link for professional visibility.")

    if not re.search(r"\S+@\S+", resume_text):
        feedback.append("Add a professional email address.")

    return feedback
model_pipeline, metadata = load_model()

# App title
st.title("AI Career Compass")
st.subheader("Student Placement Prediction and Career Guidance System")

st.write(
    "Enter student profile details in the sidebar and click the prediction button "
    "to estimate placement readiness."
)

# Sidebar inputs
st.sidebar.header("Student Profile Input")

age = st.sidebar.slider("Age", 18, 25, 21)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

college_tier = st.sidebar.selectbox(
    "College Tier",
    ["Tier 1", "Tier 2", "Tier 3"]
)

specialization = st.sidebar.selectbox(
    "Specialization",
    [
        "AI/ML",
        "Civil",
        "Electronics",
        "Computer Science",
        "Information Technology",
        "Data Science",
        "Mechanical"
    ]
)

cgpa = st.sidebar.slider("CGPA", 4.5, 10.0, 7.5, 0.1)

dsa_problems = st.sidebar.slider("DSA Problems Solved", 0, 850, 250)

internships = st.sidebar.slider("Internships", 0, 4, 1)

certifications = st.sidebar.slider("Certifications", 0, 10, 5)

projects_count = st.sidebar.slider("Projects Count", 1, 12, 6)

communication_skills = st.sidebar.slider("Communication Skills", 0, 100, 70)

aptitude_score = st.sidebar.slider("Aptitude Test Score", 0, 100, 70)

leetcode_rating = st.sidebar.slider("LeetCode Rating", 0, 2500, 1500)

github_contributions = st.sidebar.slider("GitHub Contributions", 0, 500, 100)

hackathons = st.sidebar.slider("Hackathons Participated", 0, 10, 2)

ai_ml_skill = st.sidebar.slider("AI/ML Skill Level", 0, 100, 60)

system_design = st.sidebar.slider("System Design Knowledge", 0, 100, 50)

resume_score = st.sidebar.slider("Resume Score", 0, 100, 70)

mock_interview = st.sidebar.slider("Mock Interview Score", 0, 100, 70)

# Feature engineering
academic_score = (cgpa / 10) * 100

coding_score = (
    (dsa_problems / 841) * 40 +
    (leetcode_rating / 2500) * 30 +
    (github_contributions / 500) * 30
)

experience_score = (
    (internships / 4) * 40 +
    (projects_count / 12) * 35 +
    (hackathons / 10) * 25
)

skill_score = (
    (ai_ml_skill / 100) * 50 +
    (system_design / 100) * 50
)

professional_readiness_score = (
    communication_skills * 0.25 +
    aptitude_score * 0.25 +
    resume_score * 0.25 +
    mock_interview * 0.25
)

employability_score = (
    academic_score * 0.20 +
    coding_score * 0.25 +
    experience_score * 0.20 +
    skill_score * 0.15 +
    professional_readiness_score * 0.20
)

# Create input dataframe for model
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "College_Tier": college_tier,
    "Specialization": specialization,
    "CGPA": cgpa,
    "DSA_Problems_Solved": dsa_problems,
    "Internships": internships,
    "Certifications": certifications,
    "Projects_Count": projects_count,
    "Communication_Skills": communication_skills,
    "Aptitude_Test_Score": aptitude_score,
    "LeetCode_Rating": leetcode_rating,
    "GitHub_Contributions": github_contributions,
    "Hackathons_Participated": hackathons,
    "AI_ML_Skill_Level": ai_ml_skill,
    "System_Design_Knowledge": system_design,
    "Resume_Score": resume_score,
    "Mock_Interview_Score": mock_interview,
    "Academic_Score": academic_score,
    "Coding_Score": coding_score,
    "Experience_Score": experience_score,
    "Skill_Score": skill_score,
    "Professional_Readiness_Score": professional_readiness_score,
    "Employability_Score": employability_score
}])

# Prediction section
if st.sidebar.button("Predict Placement"):
    st.session_state["show_prediction"] = True

if st.session_state.get("show_prediction", False):

    prediction = model_pipeline.predict(input_data)[0]
    probability = model_pipeline.predict_proba(input_data)[0][1]
    dashboard_tab, career_tab, resume_tab, about_tab = st.tabs(
        [
            "Dashboard",
            "Career Guidance",
            "Resume Analyzer",
            "About Project"
        ]
    )
    with dashboard_tab:
      col1, col2, col3 = st.columns(3)

      with col1:
            st.metric("Placement Probability", f"{probability * 100:.2f}%")

      with col2:
            if prediction == 1:
                  st.metric("Prediction", "Higher Readiness")
            else:
                  st.metric("Prediction", "Lower Readiness")

      with col3:
            st.metric("Employability Score", f"{employability_score:.2f}")

      st.progress(float(probability))

      st.subheader("Student Profile Insights")

      insight_data = pd.DataFrame({
            "Category": [
                  "Academic Score",
                  "Coding Score",
                  "Experience Score",
                  "Skill Score",
                  "Professional Readiness",
                  "Employability Score"
            ],
            "Score": [
                  academic_score,
                  coding_score,
                  experience_score,
                  skill_score,
                  professional_readiness_score,
                  employability_score
            ]
      })

      st.bar_chart(insight_data.set_index("Category"))

      st.subheader("Input Summary")
      st.dataframe(input_data)
    with career_tab:
      st.subheader("AI Career Guidance Module")

      preferred_role = st.selectbox(
            "Preferred Career Role",
            [
                  "Machine Learning Engineer",
                  "Data Scientist",
                  "AI Engineer",
                  "Software Engineer",
                  "Data Analyst",
                  "Full Stack Developer",
                  "Cloud Engineer",
                  "Cybersecurity Analyst"
            ]
      )

      learning_duration = st.selectbox(
            "Preferred Learning Roadmap Duration",
            [
                  "30 Days",
                  "60 Days",
                  "90 Days",
                  "6 Months"
            ]
      )

      career_prompt = f"""
      You are an expert AI career counselor, placement mentor, and technical interview coach.

      Analyze the following student profile and generate a detailed but practical career guidance report.

      Student Profile:
      - Age: {age}
      - Gender: {gender}
      - College Tier: {college_tier}
      - Specialization: {specialization}
      - CGPA: {cgpa}
      - DSA Problems Solved: {dsa_problems}
      - Internships: {internships}
      - Certifications: {certifications}
      - Projects Count: {projects_count}
      - Communication Skills Score: {communication_skills}
      - Aptitude Test Score: {aptitude_score}
      - LeetCode Rating: {leetcode_rating}
      - GitHub Contributions: {github_contributions}
      - Hackathons Participated: {hackathons}
      - AI/ML Skill Level: {ai_ml_skill}
      - System Design Knowledge: {system_design}
      - Resume Score: {resume_score}
      - Mock Interview Score: {mock_interview}

      Model Results:
      - Placement Prediction: {"Higher Placement Readiness" if prediction == 1 else "Lower Placement Readiness"}
      - Placement Probability: {probability * 100:.2f}%
      - Academic Score: {academic_score:.2f}
      - Coding Score: {coding_score:.2f}
      - Experience Score: {experience_score:.2f}
      - Skill Score: {skill_score:.2f}
      - Professional Readiness Score: {professional_readiness_score:.2f}
      - Employability Score: {employability_score:.2f}

      Student Preference:
      - Preferred Career Role: {preferred_role}
      - Roadmap Duration: {learning_duration}

      Generate the response in this exact structure:

      1. Career Fit Summary
      Explain whether the preferred role is suitable for this student.

      2. Recommended Career Paths
      Suggest 3 suitable career paths and explain why.

      3. Strengths
      List the student's strongest areas.

      4. Skill Gap Analysis
      Identify missing or weak skills.

      5. Certification Recommendations
      Suggest relevant certifications, courses, or platforms.

      6. Project Recommendations
      Suggest 3 portfolio projects that can improve placement chances.

      7. Personalized Learning Roadmap
      Create a roadmap for the selected duration.

      8. Resume Improvement Tips
      Suggest improvements based on the resume score.

      9. Interview Preparation Plan
      Give technical, aptitude, communication, and HR interview preparation advice.

      10. Final Action Plan
      Give 5 clear next actions.

      Keep the tone professional, practical, and student-friendly.
      Avoid unrealistic claims.
      Keep the response under 700 words.
      """

      if st.button("Generate Full Career Guidance"):
            with st.spinner("Generating personalized career guidance using Gemini..."):
                  career_guidance = get_gemini_response(career_prompt)
                  st.session_state["career_guidance"] = career_guidance

      if "career_guidance" in st.session_state:
            st.markdown(st.session_state["career_guidance"])
    with resume_tab:
      st.subheader("Resume Analyzer Module")

      uploaded_resume = st.file_uploader(
            "Upload Resume PDF",
            type=["pdf"]
      )

      if uploaded_resume is not None:
            resume_text = extract_text_from_pdf(uploaded_resume)

            if resume_text.startswith("PDF extraction error"):
                  st.error(resume_text)

            elif len(resume_text.strip()) == 0:
                  st.warning("No readable text found in the PDF. Please upload a text-based resume PDF.")

            else:
                  found_skills = identify_resume_skills(resume_text)
                  calculated_resume_score = calculate_resume_score(resume_text, found_skills)
                  basic_feedback = generate_basic_resume_feedback(
                  calculated_resume_score,
                  found_skills,
                  resume_text
                  )

                  col1, col2, col3 = st.columns(3)

                  with col1:
                        st.metric("Resume Score", f"{calculated_resume_score}/100")

                  with col2:
                        st.metric("Skills Found", len(found_skills))

                  with col3:
                        st.metric("Resume Words", len(resume_text.split()))

                  st.progress(calculated_resume_score / 100)

                  st.subheader("Identified Skills")

                  if found_skills:
                        st.write(", ".join(found_skills))
                  else:
                        st.warning("No major technical skills found. Add a clear Skills section.")

                  st.subheader("Resume Improvement Suggestions")

                  for suggestion in basic_feedback:
                        st.write(f"- {suggestion}")

                  with st.expander("View Extracted Resume Text"):
                        st.write(resume_text)

                  resume_prompt = f"""
                  You are an expert resume reviewer and placement mentor.

                  Review this student's resume and provide practical improvement suggestions.

                  Student Career Context:
                  - Preferred Specialization: {specialization}
                  - Placement Probability: {probability * 100:.2f}%
                  - Employability Score: {employability_score:.2f}
                  - Current Resume Score: {calculated_resume_score}/100
                  - Skills Identified: {found_skills}

                  Resume Text:
                  {resume_text[:4000]}

                  Generate output in this structure:

                  1. Resume Summary
                  2. Strong Points
                  3. Missing Skills Or Sections
                  4. Bullet Point Improvements
                  5. ATS Optimization Tips
                  6. Final Resume Action Plan

                  Keep the feedback practical and student-friendly.
                  """

                  if st.button("Generate AI Resume Feedback"):
                        with st.spinner("Analyzing resume using Gemini..."):
                              ai_resume_feedback = get_gemini_response(resume_prompt)
                              st.session_state["ai_resume_feedback"] = ai_resume_feedback

                  if "ai_resume_feedback" in st.session_state:
                        st.markdown(st.session_state["ai_resume_feedback"])
    with about_tab:
        st.subheader("About AI Career Compass")

        st.write("""
        AI Career Compass is an end-to-end machine learning and AI-powered
        placement prediction and career guidance platform for students.

        The system uses a Random Forest Classifier to estimate placement readiness
        and Google Gemini API to generate personalized career guidance and resume feedback.
        """)

        st.write("Key Features:")

        st.write("""
        - Placement readiness prediction
        - Placement probability estimation
        - Student profile insights
        - AI-powered career guidance
        - Skill gap analysis
        - Personalized learning roadmap
        - PDF resume analyzer
        - AI resume improvement suggestions
        """)

        st.write("Technology Stack:")

        st.write("""
        Python, Pandas, NumPy, Scikit-Learn, Random Forest Classifier,
        Streamlit, Google Gemini API, PDFPlumber, Joblib
        """)

if not st.session_state.get("show_prediction", False):
    st.info("Enter student details from the sidebar and click Predict Placement.")