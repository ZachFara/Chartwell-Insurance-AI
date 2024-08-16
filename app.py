import streamlit as st
import openai
from pinecone import Pinecone
from PyPDF2 import PdfFileReader
import os
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key="3f6ed6fe-57ee-48af-b5b8-268b75a22022")
index = pc.Index("insurancedoc")
openai.api_key = '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'


def read_text_file(file, encoding='utf-8'):
    try:
        with open(file, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin-1') as file:
            return file.read()

def read_pdf_file(file):
    with open(file, 'rb') as file:
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

def upload_documents_to_pinecone(files):
    for file in files:
        if file.name.endswith('.txt'):
            document_text = read_text_file(file)
        elif file.name.endswith('.pdf'):
            document_text = read_pdf_file(file)
        else:
            st.warning(f"Unsupported file format: {file.name}")
            continue
        
        document_embedding = get_embeddings(document_text)
        document_id = file.name
        
        index.upsert([
            (document_id, document_embedding, {"text": document_text})
        ])
        st.success(f"Document '{document_id}' successfully added to Pinecone index.")

# Streamlit UI
st.title("Document Upload to Pinecone")
st.write("Drag and drop your text or PDF documents below to upload them to Pinecone.")

uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

if st.button("Upload"):
    if uploaded_files:
        upload_documents_to_pinecone(uploaded_files)
    else:
        st.warning("Please upload at least one document.")