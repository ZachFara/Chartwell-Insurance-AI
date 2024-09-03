import streamlit as st
import openai
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
import io
from dotenv import load_dotenv
# import nest_asyncio
# nest_asyncio.apply()
# from llama_parse import LlamaParse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.text_cleaning import remove_substrings, collapse_spaces
from utils.getting_embeddings import get_embeddings, chunk_text
from utils.querying_pinecone import retrieve_contexts, generate_response, augment_query, filter_contexts
import fitz
# Load environment variables, bring in LLAMA_CLOUD_API_KEY
load_dotenv()

primer = """You are a highly intelligent Q&A bot for Chartwell Insurance, 
designed to assist our customer service team by providing accurate and professional answers to customer queries and emails. 
Your responses should be based strictly on the information provided by the user in their query. 
Use the following pieces of context to answer the question at the end in detail with clear explanation. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Always maintain a professional and courteous tone, as if you are representing Chartwell Insurance.
"""

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("insurancedoc")

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_text_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def read_pdf_file(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return None

def clean_text(text):
    if text is None:
        return None
    text = remove_substrings(text, ["/C20", "\n"], " ")
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
        
        # Debugging: Print the length of the document text
        print(f"Length of document text: {len(document_text)}")
        
        text_chunks = chunk_text(document_text)  # Get the text chunks
        embeddings = get_embeddings(text_chunks)  # Get embeddings for each chunk
        document_id = os.path.basename(file_path)
        
        # Debugging: Print the number of chunks and embeddings
        print(f"Number of chunks: {len(text_chunks)}")
        print(f"Number of embeddings: {len(embeddings)}")
        
        # Upsert each embedding into Pinecone
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
            print(f"Uploading chunk {i} for document {document_id}")
            index.upsert([(f"{document_id}_chunk_{i}", embedding, {"text": chunk})])
        
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

def upload_documents_to_pinecone(file_paths):
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_document, file_path): file_path for file_path in file_paths}
        for future in as_completed(future_to_file):
            error, success = future.result()
            results.append((error, success))
    return results

def query_pinecone(query):
    query_embedding = get_embeddings(query, openai)[0]
    
    contexts = retrieve_contexts(index, query_embedding, 10)

    # No filtering for now.    
    
    augmented_query = augment_query(query, contexts)
    
    response = generate_response(primer, augmented_query, openai)
        
    return response
st.image("https://www.chartwellins.com/img/~www.chartwellins.com/layout-assets/logo.png")
st.title("Chartwell Insurance AI Database")

st.header("Document Upload")
uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)
# st.markdown('''Currently, we support the upload of 1,000 pages per day (1200 pages per file max)''')

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
        progress_bar = st.progress(0)
        answer = query_pinecone(user_query)
        progress_bar.progress(50)
        st.markdown(answer)
        progress_bar.progress(100)