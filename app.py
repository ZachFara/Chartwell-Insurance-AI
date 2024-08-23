import streamlit as st
# from openai import OpenAI
import openai
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
import io
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.text_cleaning import remove_substrings, collapse_spaces
from utils.getting_embeddings import get_embeddings
from utils.querying_pinecone import retrieve_contexts, generate_response, augment_query, filter_contexts

# Load environment variables 
load_dotenv()

primer = """You are a Q&A bot for an insurance company - Chartwell Insurance. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information cannot be found in the information
provided by the user, you truthfully say 'I don't know'. When providing answers, your tone is like speaking for our company.
"""

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("insurancedoc")
client = os.getenv("OPENAI_API_KEY")

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



def clean_text(text):
    text = remove_substrings(text, {"/C20", "\n"}, " ")
    text = collapse_spaces(text)
    return text

def process_document(file_path):
    try:
        if file_path.endswith('.txt'):
            document_text = read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            document_text = read_pdf_file(file_path)
        else:
            return f"Unsupported file format: {file_path}", None
        
        document_text = clean_text(document_text)
        
        embeddings = get_embeddings(document_text, client)
        document_id = os.path.basename(file_path)
        
        # Upsert each embedding into Pinecone
        for i, embedding in enumerate(embeddings):
            
            print(embedding)
            
            DOCUMENT_LENGTH_LIMIT = 20_000
            
            if len(document_text) > DOCUMENT_LENGTH_LIMIT:  # Check if the concatenated text exceeds the limit
                document_text = document_text[:DOCUMENT_LENGTH_LIMIT]
            
            index.upsert([(f"{document_id}_chunk_{i}", embedding, {"text": document_text})])
        
        return None, f"Document '{document_id}' successfully added to Pinecone index."
    except Exception as e:
        return f"Error processing file {file_path}: {e}", None

def upload_documents_to_pinecone(file_paths):
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_document, file_path): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            error, success = future.result()
            results.append((error, success))
    return results

def query_pinecone(query):
    query_embedding = get_embeddings(query, client)[0]
    
    contexts = retrieve_contexts(index, query_embedding, 10)

    # No filtering for now.    
    
    augmented_query = augment_query(query, contexts)
    
    response = generate_response(primer, augmented_query, client)
        
    return response


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
            results = upload_documents_to_pinecone(file_paths)

        for error, success in results:
            if error:
                st.error(error)
            if success:
                st.success(success)
    else:
        st.error("Please upload at least one file.")

# New section to query the AI
st.header("Ask a Question")
user_query = st.text_input("Enter your question:")
if st.button("Submit Query"):
    with st.spinner('Querying the AI...'):
        answer = query_pinecone(user_query)
    st.write("**Answer:**", answer)
