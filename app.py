import streamlit as st
import openai
from pinecone import Pinecone
from docx2pdf import convert as docx_to_pdf
import os
import re
from dotenv import load_dotenv
import nest_asyncio
import time
import streamlit as st
# from bokeh.models.widgets import Button
from streamlit.components.v1 import html
# from bokeh.models import CustomJS
# from streamlit_bokeh_events import streamlit_bokeh_events
import time
nest_asyncio.apply()
from llama_parse import LlamaParse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.text_cleaning import read_text_file, clean_text
from utils.getting_embeddings import get_embeddings
from utils.querying_pinecone import retrieve_contexts, generate_response, augment_query, filter_contexts, retrieve_contexts_with_metadata
from utils.ChunkingHandler import ChunkingHandler

# Load environment variables 
load_dotenv()

primer = """You are a highly intelligent Q&A bot for Chartwell Insurance, 
designed to assist our customer service team by providing accurate and professional answers to customer queries and emails. 
Your response should be based on the information available in the documents uploaded and some reasonable conclusions that you can make from them.
Use the following pieces of context to answer the question at the end in detail with clear explanations. 
Always maintain a professional and courteous tone, as if you are representing Chartwell Insurance. 
Be concise yet thorough in your explanations.
Lastly, make sure to always follow the tone and structure of a customer service email.
Do not include email headers, greetings, or signatures in your response.
Also don't mention the provided context, just treat that as your knowledge base.
"""

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("insurancedoc")

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# # Initialize LlamaParse
llama_parser = LlamaParse(result_type="markdown")

#------------------Document Processing

def read_pdf_file(file_path):
    documents = llama_parser.load_data(file_path)
    return [doc.text for doc in documents]



def process_document(file_path, delimiters=None):
    """
    Processes a document by reading its content, splitting it into chunks based on size, overlap, and delimiters,
    generating embeddings for each chunk, and upserting them into a Pinecone index.

    Args:
        file_path (str): The path to the document file.
        delimiters (list, optional): A list of delimiter strings that indicate where to start a new chunk.
                                     Defaults to ['#'] if not provided.

    Returns:
        tuple: A tuple containing an error message (if any) and a success message.
    """
    try:
        if delimiters is None:
            delimiters = ['#']  # Default delimiter

        pdf_path = os.path.splitext(file_path)[0] + '.pdf'
        
        # Read the document based on its file extension
        if file_path.endswith('.txt'):
            document_texts = read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            document_texts = read_pdf_file(file_path)
        elif file_path.endswith('.docx'):
            docx_to_pdf(file_path, pdf_path)
            document_texts = read_pdf_file(pdf_path)
        else:
            return f"Unsupported file format: {file_path}", None

        # Regroup document texts with a newline separator
        document_texts = "\n".join(document_texts)

        
        document_id = os.path.basename(file_path)
        

        def chunk_text(text, delimiters=None, max_words=100, min_words=10):
            """
            Splits the text into chunks, each starting with a header defined by delimiters and limited by max_words.
            If a chunk is smaller than min_words, it is appended to the next chunk.

            Args:
                text (str): The text to split.
                delimiters (list, optional): A list of delimiter strings that indicate where to start a new chunk.
                                            Defaults to ['#'].
                max_words (int, optional): Maximum number of words per chunk. Defaults to 500.
                min_words (int, optional): Minimum number of words a chunk should have. Defaults to 100.

            Yields:
                str: A chunk of text starting with a delimiter and within the word limits.
            """
            if delimiters is None:
                delimiters = ['#']  # Default delimiter

            # Create a regex pattern to match headers starting with delimiters
            # e.g., '# ', '## ', '### ', etc.
            delimiter_pattern = '|'.join([fr'^{re.escape(d)}\s+' for d in delimiters])

            # Use lookahead to split while keeping the delimiter with the following text
            split_regex = f'(?m)(?=({delimiter_pattern}))'
            split_text = re.split(split_regex, text)

            current_chunk = ""
            buffer = ""  # Buffer to hold small chunks

            for part in split_text:
                if re.match(delimiter_pattern, part):
                    if current_chunk:
                        # Split the current_chunk into sub-chunks based on max_words
                        for sub_chunk in split_by_word_limit(current_chunk, max_words):
                            if buffer:
                                # Append buffer to the current sub_chunk
                                sub_chunk = buffer + ' ' + sub_chunk
                                buffer = ""
                            
                            word_count = len(sub_chunk.split())
                            
                            if word_count >= min_words:
                                yield sub_chunk.strip()
                            else:
                                # Buffer the small chunk to append to the next one
                                buffer = sub_chunk
                    current_chunk = part  # Start new chunk with delimiter
                else:
                    current_chunk += " " + part  # Append text to the current chunk

            # Handle the last chunk
            if current_chunk:
                for sub_chunk in split_by_word_limit(current_chunk, max_words):
                    if buffer:
                        sub_chunk = buffer + ' ' + sub_chunk
                        buffer = ""
                    
                    word_count = len(sub_chunk.split())
                    
                    if word_count >= min_words:
                        yield sub_chunk.strip()
                    else:
                        buffer = sub_chunk

            # Yield any remaining buffer
            if buffer:
                yield buffer.strip()

        def split_by_word_limit(text, max_words):
            """
            Splits the text into sub-chunks each with at most max_words.

            Args:
                text (str): The text to split.
                max_words (int): Maximum number of words per sub-chunk.

            Yields:
                str: A sub-chunk of text within the word limit.
            """
            words = text.split()
            for i in range(0, len(words), max_words):
                yield ' '.join(words[i:i + max_words])

        
        for i, doc_text in enumerate(document_texts):
            doc_text = clean_text(doc_text)
            chunk_generator = chunk_text(doc_text, max_words=250, min_words=10, delimiters=delimiters)
            for c, chunk in enumerate(chunk_generator):
                embeddings = get_embeddings(chunk, openai, model="text-embedding-3-large")
                for j, embedding in enumerate(embeddings):
                    vector_id = f"{document_id}_{i}_{c}_{j}"
                    index.upsert(vectors=[{
                        "id": vector_id,
                        "values": embedding,
                        "metadata": {"text": chunk}
                    }])
        
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
def query_pinecone(query, conversation_history, similarity_threshold=0):
    query_embedding = get_embeddings(query, openai)[0]
    contexts_with_metadata = retrieve_contexts_with_metadata(index, query_embedding, 100)

    contexts = [
        match['metadata']['text'] for match in contexts_with_metadata
        if match['score'] >= similarity_threshold
    ]

    avg_score = sum(match['score'] for match in contexts_with_metadata) / len(contexts_with_metadata)

    # Logging
    with open("last_response.txt", "w") as file:
        file.write("")
    with open("last_response.txt", "a") as file:
        file.write(f"average score: {avg_score} of retrieved contexts. kept {len(contexts)}/{len(contexts_with_metadata)} .\n")

    # combine the current query with conversation history
    full_context = "\n".join(conversation_history) + "\n" + query
    augmented_query = augment_query(full_context, contexts)

    # Logging
    with open("last_response.txt", "a") as file:
        file.write(f"all contexts seen: {augmented_query}\n")
    
    response = generate_response(primer, augmented_query, openai, model="gpt-4o")
    return response

def copy_to_clipboard(text):
    # Clean up the text
    # 1. Remove the subject line
    subject_pattern = re.compile(r"^Subject:.*$", re.MULTILINE)
    text = re.sub(subject_pattern, "", text)
    # 2. Remove bolding for MD format
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # 3. Remove any spaces or newlines that come before the response
    text = text.lstrip()
    
    # Create a simple button with JavaScript functionality
    copy_button_html = f"""
        <style>
        .copy-button {{
            background-color: rgb(19, 101, 168);
            border: none;
            color: white;
            padding: 8px 16px;
            font-size: 14px;
            cursor: pointer;
            border-radius: 5px;
            width: 50px;
            height: 35px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
        }}
        </style>
        <button 
            class="copy-button"
            onclick="navigator.clipboard.writeText(`{text.replace('`', '\\`')}`)">
            Copy
        </button>
    """
    
    html(copy_button_html, height=50)

def clear_conversation():
    st.session_state.messages = []

#------------------Streamlit Interface
st.sidebar.image("https://www.chartwellins.com/img/~www.chartwellins.com/layout-assets/logo.png", use_container_width=True)
st.sidebar.title("Chartwell Insurance AI Assistant")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Document Upload", "Chatbot", "FAQ"])

if page == "Document Upload":
    st.header("📄 Document Upload")
    st.write("Upload your documents below to add them to the AI assistant's knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="You can upload multiple files at once."
    )
    st.caption("Currently, we support uploading up to **1000 pages per day** (1200 pages per file max).")

    if st.button("Upload and Index Documents"):
        if uploaded_files:
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join("/tmp", uploaded_file.name)
                with  open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
                st.success(f"Uploaded `{uploaded_file.name}`")

            with st.spinner('Uploading and indexing documents...'):
                progress_bar = st.progress(0)
                results = upload_documents_to_pinecone(file_paths)
                progress_bar.progress(100)

            for error, success in results:
                if error:
                    st.error(error)
                if success:
                    st.success(success)
        else:
            st.warning("Please upload at least one file.")

elif page == "Chatbot":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar input for email details
    subject = st.sidebar.text_input("Email Subject", value="Your Subject Here")
    recipient = st.sidebar.text_input("Recipient Name", value="Recipient")
    sender = st.sidebar.text_input("Your Name", value="Your Name")

    # Function to format assistant's response as an email
    def format_as_email(content, subject=subject, recipient=recipient, sender=sender):
        email_content = (
            f"Subject: {subject}\n\n"
            f"Dear {recipient},\n\n"
            f"{content}\n\n"
            "Best regards,  \n"  # Add two spaces before the newline
            f"{sender}"
        )
        return email_content

    # Chat container
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
                st.markdown(f"**{message['role'].capitalize()}**: {message['content']}")

    # User input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(f"**You**: {user_input}")

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Show the spinner while processing the response
            conversation_history = [msg["content"] for msg in st.session_state.messages[-5:]]

            with st.spinner('🤖 Assistant is processing your request...'):
                response = query_pinecone(user_input, conversation_history)

                full_response = ""
                for char in response:
                    full_response += char
                    time.sleep(0.002)
                    message_placeholder.markdown(full_response + "▌")
    
                # Format the response as an email
                email_ready_response = format_as_email(full_response)
                message_placeholder.markdown(email_ready_response)
            
            # Append the assistant's email-formatted response to the session messages
            st.session_state.messages.append({"role": "assistant", "content": email_ready_response})

            # Optional: Copy the email-formatted response to clipboard
            copy_to_clipboard(email_ready_response)

    # Sidebar button to clear the conversation
    if st.sidebar.button("🗑️ Clear Conversation", on_click=clear_conversation):
        st.rerun()

    # Footer with disclaimer
    def add_footer():
        st.markdown("""
        ---
        <div style='text-align: center;'>
            <p>© 2024 <strong>Chartwell Insurance</strong>. All rights reserved.</p>
            <p><i>Disclaimer: Chartwell Insurance AI is a tool and may provide inaccurate information. Always verify important details.</i></p>
        </div>
        """, unsafe_allow_html=True)

    add_footer()



elif page == "FAQ":
    st.header("Frequently Asked Questions")

    faqs = [
        {
            "question": "How do I upload a document?",
            "answer": "Go to the 'Document Upload' page and select your files to upload."
        },
        {
            "question": "What types of files are supported?",
            "answer": "Currently, we support .txt and .pdf files."
        },
        {
            "question": "How can I ask a question?",
            "answer": "Navigate to the 'Ask a Question' page and enter your query in the text area."
        },
        {
            "question": "What is the page limit for document uploads?",
            "answer": "We support uploading up to 1,000 pages per day, with a maximum of 1,200 pages per file."
        },
        # Add more FAQs as needed
    ]

    filtered_faqs = faqs

    if filtered_faqs:
        for faq in filtered_faqs:
            with st.expander(faq["question"]):
                st.write(faq["answer"])
    else:
        st.write("No FAQs found matching your search query.")
