import streamlit as st      # for web app
import PyPDF2               # for PDF text extraction
import io                   # for file handling
import os                   # for environment variables
from openai import OpenAI        # for OpenAI API interaction
from dotenv import load_dotenv     # for loading .env files

load_dotenv()  # Load environment variables from .env file
# Streamlit app configuration
st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃", layout="centered")

# App title and description
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# File uploader and job role input
uploaded_file = st.file_uploader("Upload your resume (PDF of TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're taregtting (optional)")
# Analyze button
analyze = st.button("Analyze Resume")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")
# Resume analysis and feedback generation
if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)
        
        if not file_content.strip():
            st.error("File does not have any contnet...")
            st.stop()
        # Prepare the prompt for OpenAI API
        prompt = f"""Please analyze this resume and provide constructive feedback. 
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}
        
        Resume content:
        {file_content}
        
        Please provide your analysis in a clear, structured format with specific recommendations."""
        # Call OpenAI API for analysis
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        st.markdown("### Analysis Results")
        st.markdown(response.choices[0].message.content)
    # Handle exceptions and errors
    except Exception as e:
        st.error(f"An error occured: {str(e)}")