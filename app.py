import streamlit as st
import openai
from pinecone import Pinecone
from PyPDF2 import PdfReader
import os
import io
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
from llama_parse import LlamaParse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.text_cleaning import remove_substrings, collapse_spaces
from utils.getting_embeddings import get_embeddings
from utils.querying_pinecone import retrieve_contexts, generate_response, augment_query, filter_contexts

# Load environment variables 
load_dotenv()

primer = """You are a highly intelligent Q&A bot for Chartwell Insurance, 
designed to assist our customer service team by providing accurate and professional answers to customer queries and emails. 
Your responses should be based strictly on the information provided by the user in their query and the given context. 
Use the following pieces of context to answer the question at the end in detail with clear explanations. 
If you don't know the answer, explicitly state that you don't know, and do not attempt to fabricate an answer. 
Always maintain a professional and courteous tone, as if you are representing Chartwell Insurance. 
Be concise yet thorough in your explanations.
Lastly, make sure to always follow the tone and structure of a customer service email.
If there is a name presented to you within your prompt make sure to address the email to that person.
"""

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("insurancedoc")

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize LlamaParse
llama_parser = LlamaParse(result_type="markdown")

#------------------Document Processing

def read_text_file(file_path, encoding='utf-8'):
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()
        
def read_pdf_file(file_path):
    documents = llama_parser.load_data(file_path)
    return [doc.text for doc in documents]

def clean_text(text):
    text = remove_substrings(text, {"/C20", "\n"}, " ")
    text = collapse_spaces(text)
    return text

def process_document(file_path):
    try:
        if file_path.endswith('.txt'):
            document_texts = read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            document_texts = read_pdf_file(file_path)
        else:
            return f"Unsupported file format: {file_path}", None
        
        document_id = os.path.basename(file_path)
        
        for i, document_text in enumerate(document_texts):
            document_text = clean_text(document_text)
            embeddings = get_embeddings(document_text, openai)
            
            # Upsert each embedding into Pinecone
            for j, embedding in enumerate(embeddings):
                print(embedding)
                
                DOCUMENT_LENGTH_LIMIT = 20_000
                
                if len(document_text) > DOCUMENT_LENGTH_LIMIT:  # Check if the text exceeds the limit
                    document_text = document_text[:DOCUMENT_LENGTH_LIMIT]
                
                index.upsert([(f"{document_id}_chunk_{i}_{j}", embedding, {"text": document_text})])
        
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

#------------------Querying Pinecone
def query_pinecone(query):
    query_embedding = get_embeddings(query, openai)[0]
    
    contexts = retrieve_contexts(index, query_embedding, 20)

    # No filtering for now.    
    
    augmented_query = augment_query(query, contexts)
    
    response = generate_response(primer, augmented_query, openai)
        
    return response

#------------------Streamlit Interface
st.sidebar.image("https://www.chartwellins.com/img/~www.chartwellins.com/layout-assets/logo.png")
st.sidebar.title("Chartwell Insurance AI Database")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Document Upload", "Ask a Question"])

if page == "Document Upload":
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)
    st.caption("""
    Right now, we only support upload 1000 pages per day (1200 pages per file max)""")
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
elif page == "Ask a Question":
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if st.button("Submit Query"):
        with st.spinner('Querying the AI...'):
            progress_bar = st.progress(0)
            answer = query_pinecone(user_query)
            progress_bar.progress(50)
            
            # Debug the response
            print(answer)
            
            # Sanitize the response
            sanitized_answer = answer.replace('\n', '  \n')  # Ensure newlines are treated as line breaks in Markdown
            
            st.markdown(sanitized_answer)
            progress_bar.progress(100)
