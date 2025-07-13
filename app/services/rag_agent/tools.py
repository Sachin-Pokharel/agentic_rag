from langchain_core.tools import tool

COLLECTION_NAME = "uploaded_documents"

from services.ingestion.singleton_wrapper import get_vector_store

@tool
def search_knowledge_base(query: str) -> str:
    """
    Searches the knowledge base for documents relevant to the provided query."""
    try:
        docs = get_vector_store(COLLECTION_NAME).search(query=query, k=3)
        return docs  # return Document objects, not a string
    except Exception as e:
        raise Exception(f"Search failed: {str(e)}")
    

@tool
def book_interview(receiver_email: str, user_name: str, appointment_date: str, appointment_time: str) -> str:
    """Send a confirmation email with interview details to the provided email address."""
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    GMAIL_EMAIL = os.getenv("GMAIL_USERNAME")
    GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

    EMAIL_TEMPLATE = f"""Dear {{user_name}},

Your interview has been confirmed for {{appointment_date}} at {{appointment_time}}.

What to expect:
- Personalized AI strategy discussion
- Review of your business requirements  
- Timeline and implementation approach
- Q&A session with our experts

We'll contact you soon to confirm the exact time and provide meeting details.

Regards,
Interview Team"""

    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = "Interview Confirmation"
        body = EMAIL_TEMPLATE.format(user_name=user_name, appointment_date=appointment_date, appointment_time=appointment_time)
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_EMAIL, receiver_email, msg.as_string())

        return f"Confirmation email sent to {receiver_email}"
    except Exception as e:
        return f"Failed to send confirmation email: {str(e)}"
