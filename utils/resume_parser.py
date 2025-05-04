import fitz  # PyMuPDF
import docx2txt

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_resume_text(text):
    return ' '.join(text.split())
