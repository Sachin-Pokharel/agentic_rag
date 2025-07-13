import re

def clean_page_content(text: str) -> str:
    """
    Cleans up page content by:
    - Removing excessive newlines
    - Collapsing multiple spaces
    - Stripping leading/trailing whitespace
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n+', '\n', text)

    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Strip leading/trailing spaces on each line
    lines = [line.strip() for line in text.split('\n')]
    
    # Rejoin cleaned lines
    return "\n".join(lines).strip()
