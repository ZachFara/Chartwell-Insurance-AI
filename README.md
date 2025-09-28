<h1 align="center">
Chartwell Insurance AI Assistant
</h1>

![Chartwell Insurance Logo](https://www.chartwellins.com/img/~www.chartwellins.com/layout-assets/logo.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Document Upload](#document-upload)
  - [Chatbot](#chatbot)
  - [FAQ](#faq)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)

## Introduction

The **Chartwell Insurance AI Assistant** is a Streamlit application designed to assist the customer service team of Chartwell Insurance. It leverages OpenAI's GPT-4 model to provide accurate and professional answers to customer queries and emails. The assistant can ingest company documents to enhance its knowledge base, ensuring responses are based on the most recent and relevant information.

## Features

- **Document Upload**: Upload TXT, PDF, or DOCX files to enrich the assistant's knowledge base.
- **Intelligent Chatbot**: Interact with the AI assistant to get detailed and professional responses to customer queries.
- **Contextual Understanding**: The assistant retrieves relevant information from uploaded documents using Pinecone vector embeddings.
- **Email Formatting**: Responses are formatted as customer service emails, including subject lines and personalized greetings.
- **Session Management**: Clear conversation history and manage your chat sessions.
- **FAQ Section**: Access frequently asked questions for quick guidance.


## Prerequisites

- Python 3.8 or higher
- An OpenAI API key
- A Pinecone API key
- [LlamaParser](https://github.com/llama-parser) for PDF parsing
- [Spacy](https://spacy.io/) for natural language processing

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/chartwell-insurance-ai-assistant.git
   cd chartwell-insurance-ai-assistant
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Set Up Environment Variables**

   Create a `.env` file in the project root directory and add your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

2. **Configure Pinecone**

   Initialize Pinecone in your code or as part of your environment setup:

   ```python
   import pinecone
   pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
   ```

3. **Download Spacy Model**

   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set Up LlamaParser**

   Follow the instructions on the [LlamaParser GitHub page](https://github.com/llama-parser) to install and configure.

## Running the Application

```bash
streamlit run app.py
```

*Replace `app.py` with the name of your main Python script if different.*

## Usage

### Document Upload

1. Navigate to the **Document Upload** page.
2. Click on **"Choose TXT or PDF files"** and select your documents.
3. Click **"Upload and Index Documents"** to add them to the AI assistant's knowledge base.
   - **Note**: Currently supports up to **1,000 pages per day** (1,200 pages per file max).

### Chatbot

1. Navigate to the **Chatbot** page.
2. Fill in the email details in the sidebar:
   - **Email Subject**
   - **Recipient Name**
   - **Your Name**
3. Enter your question in the chat input box.
4. The assistant will provide a response formatted as an email.
5. Use the **Copy** button to copy the response to your clipboard.
6. Clear the conversation using the **🗑️ Clear Conversation** button in the sidebar if needed.

### FAQ

Access the **FAQ** page to find answers to common questions about using the application.

## Project Structure

```
chartwell-insurance-ai-assistant/
├── app.py
├── requirements.txt
├── README.md
├── .env
└── src/
    ├── utils/
    │   ├── document_processing.py
    │   ├── embeddings.py
    │   └── chatbot_functions.py
    └── 
```

- **app.py**: Main application script.
- **requirements.txt**: Python dependencies.
- **.env**: Environment variables (not committed to version control).
- **assets/**: Images and media for the README and application.
- **src/**: Source code directory.
  - **utils/**: Utility modules for document processing, embeddings, and chatbot functions.
  - **templates/**: Template files such as the system prompt primer.



## Disclaimer

© 2024 **Chartwell Insurance**. All rights reserved.

*Disclaimer: Chartwell Insurance AI is a tool and may provide inaccurate information. Always verify important details.*


# TODO
- Make it so that the agent can start by just connecting to the vector store and not by loading anything into it
- Fix the app to interact with the agent