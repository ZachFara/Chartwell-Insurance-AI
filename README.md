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
- **Hyperparameter Tuning**: Built-in system for optimizing chunking strategies, retrieval parameters, and system prompts through systematic evaluation.
- **Modular Architecture**: Clean separation of concerns with configurable components for easy customization and testing.


## Prerequisites

- **Python 3.13** (or compatible version)
- **OpenAI API key** - For GPT-4 model access
- **Pinecone API key** - For vector storage and retrieval
- **LlamaParse API key** (optional) - For advanced PDF parsing

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ZachFara/Chartwell-Insurance-AI.git
   cd Chartwell-Insurance-AI
   ```2. **Create a Virtual Environment**

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
   LLAMAPARSE_API_KEY=your_llamaparse_api_key  # Optional
   ```

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

## Hyperparameter Tuning

The application includes a comprehensive tuning system to optimize the AI assistant's performance:

### Running Hyperparameter Tuning

```bash
python tuning/main.py
```

The tuning system will:
- Test different chunking strategies (chunk size, overlap)
- Evaluate various retrieval parameters (top-k similarity)
- Experiment with different system prompts
- Generate detailed performance metrics and results

### Tuning Results

Results are automatically saved to:
- `tuning/results/detailed_results.csv`: Individual evaluation scores
- `tuning/results/iteration_summary.csv`: Summary statistics per iteration

You can analyze these results to identify optimal parameters for your specific use case and dataset.

## Project Structure

```
Chartwell-Insurance-AI/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md
├── .env                           # Environment variables (not committed)
├── data/                          # Data storage directory
│   └── raw/                       # Raw insurance documents (PDFs)
├── src/                           # Core application source code
│   ├── __init__.py
│   ├── agent.py                   # Main AI agent orchestrator
│   ├── configuration.py          # Configuration management
│   ├── document_loader.py         # Document processing and chunking
│   ├── vector_store_manager.py    # Pinecone vector store management
│   └── components/                # Reusable components
└── tuning/                        # Hyperparameter tuning system
    ├── main.py                    # Tuning entry point
    ├── data/
    │   ├── eval/
    │   │   └── sample_questions.csv  # Evaluation questions dataset
    │   └── system_prompts/
    │       └── system_prompts.json   # System prompt variations
    ├── results/                   # Tuning results and metrics
    │   ├── detailed_results.csv   # Detailed evaluation results
    │   └── iteration_summary.csv  # Summary of tuning iterations
    └── src/                       # Tuning system components
        ├── evaluator.py           # Response evaluation logic
        ├── hyperparameter_sampler.py  # Parameter sampling
        ├── results_manager.py     # Results tracking and analysis
        └── tuning_orchestrator.py # Main tuning coordinator
```

### Key Components

- **app.py**: Main Streamlit application interface
- **src/**: Modular core application with clean separation of concerns
  - **agent.py**: AI agent with tunable parameters (chunk size, overlap, top-k retrieval)
  - **configuration.py**: Centralized configuration management
  - **document_loader.py**: Handles document ingestion and chunking strategies
  - **vector_store_manager.py**: Manages Pinecone vector store operations
- **tuning/**: Comprehensive hyperparameter optimization system
  - Evaluates different chunking methods, retrieval parameters, and system prompts
  - Tracks performance metrics across multiple iterations
  - Supports systematic optimization of the RAG pipeline
- **data/raw/**: Insurance policy documents and reference materials



## Disclaimer

© 2024 **Chartwell Insurance**. All rights reserved.

*Disclaimer: Chartwell Insurance AI is a tool and may provide inaccurate information. Always verify important details.*

