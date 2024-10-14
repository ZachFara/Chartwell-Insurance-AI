import streamlit as st
import openai
from pinecone import Pinecone
from PyPDF2 import PdfReader
from docx2pdf import convert as docx_to_pdf
import os
import io
from dotenv import load_dotenv
import nest_asyncio
import time
import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import time
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
Do not include email headers, greetings, or signatures in your response.
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
        pdf_path = file_path.rsplit('.', 1)[0] + '.pdf' # Make a destination path, this will hold the path of the pdf file if we have to perform a conversion
        if file_path.endswith('.txt'):
            document_texts = read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            document_texts = read_pdf_file(file_path)
        elif file_path.endswith('.docx'):
            docx_to_pdf(file_path, pdf_path)
            document_texts = read_pdf_file(pdf_path)
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
                    print("Metadata length limit exceeded, cutting the length short and upserting to pinecone with some text removed!")
                    document_text = document_text[:DOCUMENT_LENGTH_LIMIT]
                
                index.upsert(
                vectors=[
                    {
                        "id": f"{document_id}_chunk_{i}_{j}",
                        "values": embedding,
                        "metadata": {"text": document_text}
                    }
                ]
            )
        
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
def query_pinecone(query, conversation_history):
    query_embedding = get_embeddings(query, openai)[0]
    contexts = retrieve_contexts(index, query_embedding, 20)
    
    # Combine the current query with conversation history
    full_context = "\n".join(conversation_history) + "\n" + query
    augmented_query = augment_query(full_context, contexts)
    
    response = generate_response(primer, augmented_query, openai)
    return response

def copy_to_clipboard(text):
    # <button class="bk bk-btn bk-btn-default" type="button">Copy</button>
    
    st.markdown("""
        <style>
        .bk, .bk-btn, .bk-btn-default {
            background-color: #2A5CAA !important;
            border: none !important;
            color: white !important;
            padding: 0 !important; /* Remove extra padding */
            font-size: 14px !important;
            cursor: pointer !important;
            border-radius: 5px !important;
            width: 50px !important;  /* Set fixed width */
            height: 35px !important;  /* Set fixed height */
            display: inline-flex !important;  /* Ensure proper inline display */
            justify-content: center;  /* Center text horizontally */
            align-items: center;  /* Center text vertically */
        }
        </style>
        """, unsafe_allow_html=True)
    
    copy_button = Button(label="Copy")
    copy_button.js_on_event("button_click", CustomJS(args=dict(text=text), code="""
        navigator.clipboard.writeText(text);
    """))
    
    st.bokeh_chart(copy_button)

def clear_conversation():
    st.session_state.messages = []

#------------------Streamlit Interface
st.sidebar.image("https://www.chartwellins.com/img/~www.chartwellins.com/layout-assets/logo.png", use_column_width=True)
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
                with open(file_path, "wb") as f:
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
            "Best regards,\n"
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

                for chunk in response.split():
                    
                    # Fix some of the formatting
                    if chunk[0].isdigit() and chunk[1] == '.':
                        full_response += "\n" + chunk + " "
                    elif chunk.startswith("- "):
                        full_response += "\n  " + chunk + " "
                    elif full_response.endswith("\n  ") and chunk[0].isdigit() and chunk[1] == '.':
                        full_response += "  " + chunk + " "
                    elif full_response.endswith("\n") and chunk.startswith("- "):
                        full_response += "  " + chunk + " "
                    else:
                        full_response += chunk + " "
                        
                    time.sleep(0.05)
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