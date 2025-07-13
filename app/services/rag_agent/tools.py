from langchain_core.tools import tool
import os 
import openai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

COLLECTION_NAME = "uploaded_documents"

from services.ingestion.singleton_wrapper import get_vector_store

@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the knowledge base for documents relevant to the provided query."""
    try:
        docs = get_vector_store(COLLECTION_NAME).search_with_scores(query=query, k=5)
        return docs  # return Document objects, not a string
    except Exception as e:
        raise Exception(f"Search failed: {str(e)}")
    

@tool
def book_interview(receiver_email: str, user_name: str, appointment_date: str, appointment_time: str) -> str:
    """Send a confirmation email with interview details to the provided email address, using an LLM-generated body."""
    # Credentials and configuration
    GMAIL_EMAIL = os.getenv("GMAIL_USERNAME")
    GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Prepare the input prompt for the LLM
    prompt = f"""
Write a professional confirmation email to {user_name}, who has scheduled an interview on {appointment_date} at {appointment_time}.
Include the following points:
- Confirm the appointment date and time
- No subject line
- Close with a warm and professional tone, signing off as "Interview Team" only (no placeholders or other names)
"""

    try:
        # Generate the email body with OpenAI
        openai.api_key = OPENAI_API_KEY
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that writes professional emails."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
        email_body = response.choices[0].message.content.strip()

        # Compose the email
        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = "Interview Confirmation"
        msg.attach(MIMEText(email_body, 'plain'))

        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_EMAIL, receiver_email, msg.as_string())

        return f"Confirmation email sent to {receiver_email}"

    except Exception as e:
        return f"Failed to send confirmation email: {str(e)}"

