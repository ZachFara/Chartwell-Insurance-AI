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

# def read_text_file(file):
#     try:
#         return file.read().decode('utf-8')
#     except UnicodeDecodeError:
#         return file.read().decode('latin-1')
    
def read_text_file(file, encoding='utf-8'):
    try:
        with open(file, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='latin-1') as file:
            return file.read()    

def read_pdf_file(file):
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

def upload_document_to_pinecone(file, file_name):
    if file_name.endswith('.txt'):
        document_text = read_text_file(file)
    elif file_name.endswith('.pdf'):
        document_text = read_pdf_file(file)
    else:
        st.error(f"Unsupported file format: {file_name}")
        return
    
    document_embedding = get_embeddings(document_text)
    document_id = file_name
    
    # Upsert data with metadata
    index.upsert([
            (document_id, document_embedding, {"text": document_text})
        ])
    st.success(f"Document '{document_id}' successfully added to Pinecone index.")

st.title("Document Upload to Pinecone")

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['txt', 'pdf'])

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        upload_document_to_pinecone(uploaded_file, file_name)