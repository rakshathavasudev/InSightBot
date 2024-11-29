import os

def process_text_file(file_path):
    """
    Process a text file by reading its contents and cleaning the text.
    
    Args:
        file_path (str): Path to the text file.
    
    Returns:
        str: Cleaned text content.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.endswith(".txt"):
        raise ValueError("Only text files (.txt) are supported.")
    
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # Clean the content
    cleaned_content = clean_text(content)
    return cleaned_content

def clean_text(text):
    """
    Clean the given text by removing unwanted characters and formatting.
    
    Args:
        text (str): Input text to clean.
    
    Returns:
        str: Cleaned text.
    """
    # Example cleaning logic
    text = text.strip()  # Remove leading/trailing whitespaces
    text = text.replace("\n", " ")  # Replace newlines with spaces
    text = " ".join(text.split())  # Remove extra spaces
    return text
