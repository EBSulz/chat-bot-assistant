# Streamlit RAG Chatbot

A chatbot that answers questions about your biographical document (life, academic track, and work track) using Retrieval-Augmented Generation (RAG) to prevent hallucination.

## Features

- **RAG-based responses**: Answers are grounded in your provided document
- **Anti-hallucination**: Strict prompting ensures no extra information is generated
- **Source citation**: See which document chunks were used for each answer
- **Multiple file formats**: Supports .txt, .md, and .pdf files
- **Streamlit UI**: Clean and interactive chat interface

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

#### Local Development

Create a `.streamlit/secrets.toml` file in the project root:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

#### Streamlit Cloud Deployment

1. Go to your Streamlit Cloud app settings
2. Navigate to "Secrets" section
3. Add the following:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

### 3. Add Your Document

Place your biographical document (life, academic track, work track) in the `documents/` directory. Supported formats:
- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF files

You can also upload a document through the Streamlit app interface.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and branch
6. Set the main file path to `app.py`
7. Add your OpenAI API key in the Secrets section (see step 2 above)
8. Click "Deploy"

## How It Works

1. **Document Processing**: Your document is loaded and split into smaller chunks
2. **Embedding**: Each chunk is converted to a vector embedding using OpenAI's embedding model
3. **Vector Store**: Embeddings are stored in a FAISS vector database for fast similarity search
4. **Query Processing**: When you ask a question:
   - Your query is embedded
   - Similar document chunks are retrieved via similarity search
   - The retrieved chunks are used as context for the LLM
5. **Response Generation**: OpenAI GPT generates an answer based strictly on the retrieved context

## Usage Tips

- Be specific in your questions for better results
- The chatbot will only answer based on information in your document
- If information isn't in the document, the bot will explicitly state so
- Source chunks are displayed below each answer for transparency

## Troubleshooting

- **API Key Error**: Make sure your OpenAI API key is correctly set in `.streamlit/secrets.toml`
- **Document Not Found**: Ensure your document is in the `documents/` directory or upload it through the app
- **Rate Limits**: If you hit OpenAI rate limits, wait a moment and try again

