import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the app title
st.title("Resume Job Role Suggester")
st.write("Upload your resume, and the app will suggest the most suitable job role with a percentage score.")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

# Predefined job roles and their keywords
job_roles = {
    "Data Scientist": ["python", "machine learning", "data analysis", "tensorflow", "pandas", "numpy"],
    "Software Engineer": ["java", "c++", "software development", "git", "agile", "debugging"],
    "Web Developer": ["html", "css", "javascript", "react", "node.js", "web development"],
    "DevOps Engineer": ["aws", "docker", "kubernetes", "ci/cd", "terraform", "linux"],
    "Product Manager": ["product management", "agile", "scrum", "stakeholder management", "roadmap", "jira"],
    "Graphic Designer": ["photoshop", "illustrator", "graphic design", "typography", "adobe creative suite"],
    "Marketing Manager": ["marketing strategy", "digital marketing", "seo", "social media", "campaign management"],
    "Financial Analyst": ["financial modeling", "excel", "budgeting", "forecasting", "data analysis"],
    "HR Manager": ["recruitment", "employee relations", "performance management", "hr policies", "training and development"],
    "Sales Executive": ["sales strategy", "customer relationship", "negotiation", "lead generation", "market research"],
    "Network Security Engineer": ["VPN", "", "Network Security", "Data Encryption", "Cyber Security"]
}

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + " "
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    return text

# Function to check if the document is a resume
def is_resume(text):
    # Common resume keywords
    resume_keywords = ["skills", "experience", "education", "work history", "projects", "certifications"]
    text = text.lower()
    for keyword in resume_keywords:
        if keyword in text:
            return True
    return False

# Function to suggest the best job role
def suggest_best_job_role(resume_text, job_roles):
    # Preprocess resume text
    resume_text = preprocess_text(resume_text)
    
    # Calculate similarity between resume and job roles
    best_role = None
    best_score = 0
    
    for role, keywords in job_roles.items():
        # Create a document combining resume and job keywords
        documents = [resume_text, " ".join(keywords)]
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        # Calculate cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        
        # Update best role if current score is higher
        if similarity_score > best_score:
            best_score = similarity_score
            best_role = role
    
    # Convert similarity score to percentage
    best_score_percentage = round(best_score * 100, 2)
    return best_role, best_score_percentage

if uploaded_file:
    # Extract text from resume
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format.")
    
    # Check if the document is a resume
    if not is_resume(resume_text):
        st.warning("This document does not appear to be a resume. Please upload a valid resume.")
    else:
        # Suggest the best job role
        best_role, best_score_percentage = suggest_best_job_role(resume_text, job_roles)
        
        # Display result
        st.write(f"Suggested Job Role: **{best_role}**")
        st.write(f"Match Score: **{best_score_percentage}%**")
else:
    st.warning("Please upload your resume.")