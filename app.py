import os
from tkinter import Tk, Label, Button, filedialog
from tkinter import messagebox  # For error messages
from pinecone import Pinecone
import openai
from PyPDF2 import PdfFileReader
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key="3f6ed6fe-57ee-48af-b5b8-268b75a22022")
index = pc.Index("insurancedoc")
openai.api_key = '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'

# --- Functions ---
def read_text_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def read_pdf_file(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfFileReader(file)
        text = ""
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
        return text

def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def upload_document_to_pinecone(file_path):
    """Processes and uploads a single document to Pinecone."""

    if file_path.endswith('.txt'):
        document_text = read_text_file(file_path)
    elif file_path.endswith('.pdf'):
        document_text = read_pdf_file(file_path)
    else:
        messagebox.showerror("Error", f"Unsupported file format: {file_path}")
        return

    document_embedding = get_embeddings(document_text)
    document_id = os.path.splitext(os.path.basename(file_path))[0] 

    index.upsert([
        (document_id, document_embedding, {"text": document_text})
    ])
    messagebox.showinfo("Success", f"Document '{document_id}' added to Pinecone.")

def browse_and_upload():
    """Opens a file dialog, selects a file, and uploads it."""

    file_path = filedialog.askopenfilename(
        initialdir="/", 
        title="Select a File",
        filetypes=(("Text files", "*.txt"), ("PDF files", "*.pdf"))
    )
    if file_path: 
        upload_document_to_pinecone(file_path)

# --- GUI Setup ---
window = Tk()
window.title("Insurance Document Uploader")

label = Label(window, text="Select an insurance document:")
label.pack(pady=10)

upload_button = Button(window, text="Browse & Upload", command=browse_and_upload)
upload_button.pack()

window.mainloop()