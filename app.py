import streamlit as st
import openai
from pinecone import Pinecone
from PyPDF2 import PdfFileReader
import os
import io
from dotenv import load_dotenv

def read_text_file(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')

def read_pdf_file(file):
    reader = PdfFileReader(file)
    text = ""
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        text += page.extract_text()
    return text

def get_embeddings(text, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def upload_documents_to_pinecone(files, pinecone_api_key, index_name, openai_api_key):
    Pinecone.init(api_key=pinecone_api_key)
    index = Pinecone.Index(index_name)
    
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.txt':
            document_text = read_text_file(file)
        elif file_extension == '.pdf':
            document_text = read_pdf_file(file)
        else:
            st.error(f"Unsupported file format: {file.name}")
            continue
        
        document_embedding = get_embeddings(document_text, openai_api_key)
        document_id = file.name
        
        index.upsert([
            (document_id, document_embedding, {"text": document_text})
        ])
        st.success(f"Document '{document_id}' successfully added to Pinecone index.")

# Streamlit UI
st.title("Document Upload to Pinecone")
st.write("Drag and drop your text or PDF documents below to upload them to Pinecone.")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
pinecone_api_key = st.text_input("Enter your Pinecone API Key", type="password")
index_name = st.text_input("Enter your Pinecone Index Name")

uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

if st.button("Upload"):
    if not openai_api_key or not pinecone_api_key or not index_name:
        st.warning("Please provide all the required API keys and index name.")
    elif uploaded_files:
        upload_documents_to_pinecone(uploaded_files, pinecone_api_key, index_name, openai_api_key)
    else:
        st.warning("Please upload at least one document.")
