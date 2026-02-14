# AI-Powered YouTube Summarizer

This project is a simple web application using **Gradio** and **IBM Watsonx AI** to summarize YouTube video content and allow users to ask questions related to that video.

## Features

-   **Video Summarization**: Automatically retrieves YouTube video transcripts and generates a concise summary.
-   **Q&A**: Allows users to ask questions about the video content and receive answers based on the transcript, using RAG (Retrieval-Augmented Generation) technique.

## Technologies Used

-   **Python 3.12+**
-   **Gradio**: For building the web user interface.
-   **IBM Watsonx.ai**: Provides the Large Language Model (LLM) `ibm/granite-3-2-8b-instruct` and embedding model `ibm/slate-30m-english-rtrvr-v2`.
-   **LangChain**: Framework for connecting LLMs and processing data flows.
-   **FAISS**: Vector database for semantic storage and search.
-   **YouTube Transcript API**: For fetching transcripts from YouTube videos.

## Installation Instructions

### 1. Create and activate virtual environment

It is recommended to use a virtual environment to manage dependencies.

**On Windows:**

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*Note: If `requirements.txt` is missing, you can install the libraries manually:*

```bash
pip install gradio ibm-watsonx-ai langchain-ibm langchain-community faiss-cpu youtube-transcript-api langchain
```

## Configuration

The project requires credentials from IBM Cloud to use Watsonx AI. You need to set the following environment variables:

1.  **IBM_API_KEY**: API Key of your IBM Cloud account.
2.  **IBM_PROJECT_ID**: ID of the Watsonx project you are using.

You can set environment variables in the terminal before running the application:

**Windows (PowerShell):**

```powershell
$env:IBM_API_KEY = "your_api_key_here"
$env:IBM_PROJECT_ID = "your_project_id_here"
```

**macOS/Linux:**

```bash
export IBM_API_KEY="your_api_key_here"
export IBM_PROJECT_ID="your_project_id_here"
```

Alternatively, you can create a `.env` file (if the project supports `python-dotenv`) to store these variables.

## Usage Guide

1.  After installation and configuration, run the application using the command:

    ```bash
    python ytbot.py
    ```

2.  The application will launch and provide a local URL (usually `http://0.0.0.0:7860` or `http://127.0.0.1:7860`).

3.  Open your browser and navigate to the provided URL.

4.  **How to use:**
    -   Enter the YouTube video URL in the "YouTube Video URL" box.
    -   Click **Summarize Video** to view the summary.
    -   Enter your question in the "Ask a Question About the Video" box and click **Ask a Question** to get an answer.

## Notes

-   YouTube videos must have **English captions** for the tool to work correctly.
-   Ensure your IBM Cloud account has sufficient permissions and quota to use Watsonx models.
