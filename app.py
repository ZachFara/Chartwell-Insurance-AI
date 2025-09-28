import streamlit as st
import re
import time
from pathlib import Path
import tempfile
from src.agent import Agent

# Page config
st.set_page_config(
    page_title="Chartwell Insurance AI Assistant",
    page_icon="🏢",
    layout="wide"
)

@st.cache_resource
def initialize_agent(
    chunk_size=512, 
    chunk_overlap=50, 
    similarity_top_k=5, 
    system_prompt_override=None
):
    """Initialize the agent and connect to existing Pinecone index."""
    agent = Agent(
        name="Chartwell Insurance Assistant", 
        use_pinecone=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_top_k=similarity_top_k,
        system_prompt_override=system_prompt_override
    )
    
    # Try to connect to existing index first
    try:
        agent.connect_to_existing_index()
        st.session_state.agent_status = "Connected to existing index"
        return agent
    except Exception as e:
        st.session_state.agent_status = f"Failed to connect to existing index: {str(e)}"
        return agent

def copy_to_clipboard(text):
    """Create a copy button for text content."""
    # Clean up the text
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove markdown bold
    text = text.lstrip()
    
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
            width: 80px;
            height: 35px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
        }}
        </style>
        <button 
            class="copy-button"
            onclick='navigator.clipboard.writeText(`{text}`)'>
            Copy
        </button>
    """
    st.components.v1.html(copy_button_html, height=50)

def format_as_email(content, subject="Insurance Inquiry", recipient="Customer", sender="Chartwell Team"):
    """Format AI response as a professional email."""
    email_content = f"""Subject: {subject}

Dear {recipient},

{content}

Best regards,
{sender}
Chartwell Insurance"""
    return email_content

def clear_conversation():
    """Clear the conversation history."""
    st.session_state.messages = []

# Initialize agent
if 'agent' not in st.session_state:
    st.session_state.agent = initialize_agent()

# Sidebar
st.sidebar.image("https://www.chartwellins.com/img/~www.chartwellins.com/layout-assets/logo.png", use_container_width=True)
st.sidebar.title("Chartwell Insurance AI Assistant")

# Display agent status
if 'agent_status' in st.session_state:
    if "Connected" in st.session_state.agent_status:
        st.sidebar.success(st.session_state.agent_status)
    else:
        st.sidebar.error(st.session_state.agent_status)

# Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Document Upload", "Index Status", "FAQ"])

# Document Upload Page
if page == "Document Upload":
    st.header("📄 Document Upload")
    st.write("Upload documents to enhance the AI assistant's knowledge base.")

    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        help="Upload multiple files to add to the knowledge base."
    )

    if st.button("Upload and Index Documents", type="primary"):
        if uploaded_files:
            # Create temporary directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = Path(temp_dir) / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))
                    st.success(f"Saved {uploaded_file.name}")

                # Ingest documents using the agent
                with st.spinner('Processing and indexing documents...'):
                    progress_bar = st.progress(0)
                    
                    try:
                        st.session_state.agent.ingest_directory(temp_dir)
                        progress_bar.progress(100)
                        st.success("✅ Documents successfully indexed!")
                        
                        # Update status
                        st.session_state.agent_status = f"Index updated with {len(uploaded_files)} new documents"
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error indexing documents: {str(e)}")
                        progress_bar.progress(0)
        else:
            st.warning("Please upload at least one file.")

# Chatbot Page
elif page == "Chatbot":
    st.header("💬 AI Assistant")
    
    # Check if agent is ready
    if st.session_state.agent.agent is None:
        st.warning("⚠️ Agent not connected to index. Please upload documents or check your Pinecone configuration.")
        st.stop()

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Email formatting options in sidebar
    st.sidebar.subheader("📧 Email Formatting")
    subject = st.sidebar.text_input("Subject", value="Insurance Inquiry")
    recipient = st.sidebar.text_input("Recipient", value="Customer")
    sender = st.sidebar.text_input("Sender", value="Chartwell Team")
    format_email = st.sidebar.checkbox("Format as Email", value=True)

    # Chat interface
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
                st.markdown(message['content'])

    # User input
    if user_input := st.chat_input("Ask about insurance policies, coverage, claims..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            with st.spinner('🤖 Analyzing documents and generating response...'):
                try:
                    # Get response from agent
                    response = st.session_state.agent.chat(user_input)
                    
                    # Format as email if requested
                    if format_email:
                        response = format_as_email(response, subject, recipient, sender)
                    
                    # Simulate typing effect
                    full_response = ""
                    for char in response:
                        full_response += char
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add to conversation history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Copy button
                    copy_to_clipboard(full_response)
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")

    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation", on_click=clear_conversation):
        st.rerun()

# Index Status Page
elif page == "Index Status":
    st.header("📊 Index Status")
    
    # Get agent stats
    stats = st.session_state.agent.get_index_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Agent Information")
        st.write(f"**Agent Name:** {stats.get('agent_name', 'N/A')}")
        st.write(f"**Storage Type:** {stats.get('storage_type', 'N/A')}")
        st.write(f"**Has Index:** {'✅ Yes' if stats.get('has_index') else '❌ No'}")
        st.write(f"**Agent Ready:** {'✅ Yes' if stats.get('has_agent') else '❌ No'}")
    
    with col2:
        st.subheader("Configuration")
        st.write(f"**Pinecone Configured:** {'✅ Yes' if stats.get('pinecone_configured') else '❌ No'}")
        st.write(f"**Config Status:** {stats.get('config_status', 'N/A')}")
    
    # Agent actions
    st.subheader("Actions")
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🔄 Reconnect to Index"):
            try:
                result = st.session_state.agent.connect_to_existing_index()
                st.success(result)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to reconnect: {str(e)}")
    
    with col4:
        if st.button("🗑️ Reset Agent"):
            st.session_state.agent.reset()
            st.session_state.messages = []
            st.success("Agent reset successfully!")
            st.rerun()

# FAQ Page
elif page == "FAQ":
    st.header("❓ Frequently Asked Questions")

    faqs = [
        {
            "question": "How does the AI assistant work?",
            "answer": "The AI assistant uses advanced language models and retrieves information from indexed insurance documents to provide accurate, contextual responses to your queries."
        },
        {
            "question": "What documents are in the knowledge base?",
            "answer": "The knowledge base contains various insurance policies, contracts, and documentation including Berkley One policies, Chubb contracts, and other insurance-related materials."
        },
        {
            "question": "How do I upload new documents?",
            "answer": "Go to the 'Document Upload' page, select your PDF or TXT files, and click 'Upload and Index Documents'. The AI will process and add them to its knowledge base."
        },
        {
            "question": "Can I format responses as emails?",
            "answer": "Yes! In the Chatbot page, use the sidebar options to customize email formatting including subject, recipient, and sender information."
        },
        {
            "question": "What if the assistant can't find relevant information?",
            "answer": "The assistant will let you know if it cannot find relevant information in the indexed documents. You may need to upload additional documents or rephrase your question."
        },
        {
            "question": "How do I clear my conversation history?",
            "answer": "Use the '🗑️ Clear Conversation' button in the sidebar of the Chatbot page to start a fresh conversation."
        }
    ]

    for faq in faqs:
        with st.expander(faq["question"]):
            st.write(faq["answer"])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p>© 2024 <strong>Chartwell Insurance</strong>. All rights reserved.</p>
    <p><i>Disclaimer: This AI assistant is a tool and may provide inaccurate information. Always verify important details with official documentation.</i></p>
</div>
""", unsafe_allow_html=True)
