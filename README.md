# 🌌 PDF RAG Chatbot

A cosmic-themed PDF chatbot powered by LangChain, Chroma, and Google Gemini AI.

## Features

- 📄 Upload and index PDF documents
- 🔍 Semantic search across PDF content
- 💬 Conversational AI responses using Google Gemini
- 📚 Automatic chat history saving and loading
- 📌 Source page tracking
- 🌌 Beautiful cosmic UI theme

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create `.env` File

Copy `.env.example` to `.env` and add your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://aistudio.google.com/apikey

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Upload a PDF** - Click the upload box in the main area
2. **Index the PDF** - Wait for the indexing to complete
3. **Ask Questions** - Type your questions about the PDF content
4. **View Sources** - Click "Sources" to see which pages contain the answers
5. **Chat History** - Your conversations are automatically saved

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── test.py               # Test file
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Environment variables (not in git)
├── .gitignore           # Git ignore rules
└── chat_history/        # Saved conversations (auto-created)
```

## Chat History

Conversations are automatically saved in the `chat_history/` directory as JSON files. Each PDF gets its own conversation history.

## Troubleshooting

- **API Key Error**: Ensure `.env` file exists and contains `GOOGLE_API_KEY`
- **PDF Indexing Slow**: Larger PDFs take longer to index. This is normal.
- **Model Not Available**: Try switching to `gemini-2.0-flash` in the Settings

## Notes

- This app uses the newer `google.genai` package instead of the deprecated `google.generativeai` package.
- Chat histories are stored locally in JSON format
- The cosmic theme is optimized for dark mode
