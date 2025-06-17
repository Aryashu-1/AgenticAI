import warnings
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fpdf import FPDF
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, MDXSearchTool
from crewai.tools import BaseTool
from langchain_groq import ChatGroq
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import streamlit as st
from dotenv import load_dotenv  
load_dotenv()

# ========== PDF Reader ===========s
from typing import Optional
from pydantic import Field

class PDFReadTool(BaseTool):
    name: str = Field(default="PDFReadTool", description="Tool name")
    description: str = Field(default="Reads and extracts text from a PDF file.", description="Tool description")
    file_path: Optional[str] = None

    def _run(self, *args, **kwargs):
        text = ""
        with fitz.open(self.file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text


# ========== Markdown to PDF ===========
def markdown_to_pdf(md_path, pdf_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)

    for line in lines:
        pdf.multi_cell(0, 10, line.strip())

    pdf.output(pdf_path)

class SimpleSemanticSearch:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, small model

        # Load and split text from PDF
        self.text_chunks = self._load_and_chunk_pdf()

        # Create embeddings for chunks
        self.embeddings = self.model.encode(self.text_chunks, convert_to_numpy=True)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def _load_and_chunk_pdf(self):
        with fitz.open(self.pdf_path) as pdf:
            full_text = ""
            for page in pdf:
                full_text += page.get_text()

        # Simple chunking by splitting every 500 characters
        chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]
        return chunks

    def query(self, query_text, top_k=3):
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])
        return results

# ========== Email Utility ===========
def send_email_with_pdfs(receiver_email, subject, body, files, sender_email, sender_password):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    for file in files:
        attachment = open(file, 'rb')
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(file)}")
        msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()

# ========== CrewAI Pipeline ===========
def run_job_pipeline(job_posting_url, github_url, personal_writeup, resume_pdf_path, role, sender_email, sender_password, receiver_email):
    warnings.filterwarnings('ignore')

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()
    read_resume = PDFReadTool(file_path=resume_pdf_path)
    semantic_search_resume = SimpleSemanticSearch(resume_pdf_path)


    # Agents
    researcher = Agent(
        role=f"{role} Job Market Analyst",
        goal=f"Thoroughly analyze job postings for {role} roles to extract actionable insights, skills, and requirements.",
        tools=[scrape_tool, search_tool],
        verbose=True,
        llm=llm,
        backstory="You're a job intelligence analyst focused on decoding complex job descriptions and market trends. Your mission is to provide deep, structured insights into employer expectations so job seekers can precisely tailor their applications."
    )

    resume_strategist = Agent(
        role="Strategic Resume Architect",
        goal=f"Craft a highly targeted and compelling resume tailored specifically for the {role} role.",
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=True,
        llm=llm,
        backstory="You are a resume optimization expert. Your skill lies in aligning personal strengths and experiences with job role demands, optimizing resumes to beat ATS filters and captivate recruiters."
    )

    profiler = Agent(
        role="Candidate Intelligence Profiler",
        goal="Synthesize a technical and personal profile that reflects the applicant’s capabilities, interests, and communication style.",
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=True,
        llm=llm,
        backstory="You excel in profiling professionals using data from GitHub, resumes, and write-ups. Your insights add depth to resumes and help reveal the unique edge of each candidate."
    )

    interview_preparer = Agent(
        role="AI-Powered Interview Coach",
        goal="Generate insightful interview questions and preparation notes that help candidates confidently present their strengths.",
        tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
        verbose=True,
        llm=llm,
        backstory="You're an AI coach skilled in anticipating role-specific interview scenarios. You create intelligent question sets and prep material to help applicants shine in technical and behavioral interviews."
    )

    # Tasks
    research_task = Task(
        description=("Analyze the job posting at {job_posting_url} for key requirements."),
        expected_output="Structured job requirements.",
        agent=researcher,
        async_execution=True
    )

    profile_task = Task(
        description=("Generate profile from GitHub {github_url} and write-up {personal_writeup} ."),
        expected_output="Detailed candidate profile.",
        agent=profiler,
        async_execution=True
    )

    resume_task = Task(
        description="Refine resume based on job and profile.",
        expected_output="Tailored resume markdown.",
        output_file="tailored_resume.md",
        context=[research_task, profile_task],
        agent=resume_strategist
    )

    interview_task = Task(
        description="Create interview preparation guide.",
        expected_output="Questions and talking points.",
        output_file="interview_materials.md",
        context=[research_task, profile_task, resume_task],
        agent=interview_preparer
    )

    # Crew
    job_crew = Crew(
        agents=[researcher, profiler, resume_strategist, interview_preparer],
        tasks=[research_task, profile_task, resume_task, interview_task],
        verbose=True
    )

    job_crew.kickoff(inputs={
        "job_posting_url": job_posting_url,
        "github_url": github_url,
        "personal_writeup": personal_writeup
    })

    # Convert outputs to PDF
    markdown_to_pdf("tailored_resume.md", "tailored_resume.pdf")
    markdown_to_pdf("interview_materials.md", "interview_materials.pdf")

    # Send email
    send_email_with_pdfs(
        receiver_email=receiver_email,
        subject="Your Job Application Materials",
        body="Please find your tailored resume and interview materials attached.",
        files=["tailored_resume.pdf", "interview_materials.pdf"],
        sender_email=sender_email,
        sender_password=sender_password
    )

# ========== Streamlit UI ===========

st.title("AI Job Application Assistant")



job_posting_url = st.text_input(
    "Job Posting URL", 
    value="https://www.naukri.com/job-listings-fresher-unifybrains-infotech-pvt-ltd-kolkata-mumbai-new-delhi-hyderabad-pune-chennai-bengaluru-0-to-2-years-200520500046?src=seo_srp&sid=17494841575925399&xp=1&px=1"
)

github_url = st.text_input(
    "GitHub URL", 
    value="https://github.com/Aryashu-1"
)

personal_writeup = st.text_area(
    "Personal Writeup", 
    value=(
        "Curious by nature and driven by purpose, I constantly seek to learn and grow.\n"
        "Blending creativity with logic, I thrive at the intersection of ideas and execution.\n"
        "Passionate about using technology for meaningful impact, I build with intention.\n"
        "Grounded by values, I believe in consistency, kindness, and lifelong learning."
    )
)

role = st.text_input("Target Role", value="Software Engineer")

sender_email = st.text_input("Sender Gmail Address", value="monkeydluffy4969@gmail.com")

sender_password = st.text_input("Sender Gmail App Password", type="password", value="Arya@2004")

receiver_email = st.text_input("Receiver Email", value="aryataduri@gmail.com")

resume_pdf = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])


if st.button("Generate & Send PDFs"):
    if all([job_posting_url, github_url, personal_writeup, resume_pdf, role, sender_email, sender_password, receiver_email]):
        with open("uploaded_resume.pdf", "wb") as f:
            f.write(resume_pdf.read())

        run_job_pipeline(
            job_posting_url=job_posting_url,
            github_url=github_url,
            personal_writeup=personal_writeup,
            resume_pdf_path="uploaded_resume.pdf",
            role=role,
            sender_email=sender_email,
            sender_password=sender_password,
            receiver_email=receiver_email
        )

        st.success("PDFs generated and emailed successfully!")
    else:
        st.error("Please fill all fields and upload resume.")
