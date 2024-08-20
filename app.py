import streamlit as st
import openai
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
import io
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables 
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("insurancedoc")
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_text_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def read_pdf_file(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def upload_document_to_pinecone(file_path):
    try:
        if file_path.endswith('.txt'):
            document_text = read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            document_text = read_pdf_file(file_path)
        else:
            st.error(f"Unsupported file format: {file_path}")
            return
        
        document_embedding = get_embeddings(document_text)
        document_id = os.path.basename(file_path)
        
        index.upsert([
            (document_id, document_embedding, {"text": document_text})
        ])
        st.success(f"Document '{document_id}' successfully added to Pinecone index.")
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")

def upload_documents_to_pinecone(file_paths):
    with ThreadPoolExecutor() as executor:
        executor.map(upload_document_to_pinecone, file_paths)

st.title("Document Upload for Chartwell Insurance AI Database")

uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

if st.button("Upload and Index Documents"):
    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("/tmp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        with st.spinner('Uploading and indexing documents...'):
            upload_documents_to_pinecone(file_paths)
    else:
        st.error("Please upload at least one file.")