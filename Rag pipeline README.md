# RAG Pipeline on a Phone

A local Retrieval-Augmented Generation (RAG) pipeline running entirely on an Android phone via Termux, Gemma 4, and Ollama.

## What It Does

- Takes any PDF document (lecture notes, contracts, textbooks)
- Extracts and chunks the text
- Creates local vector embeddings
- Answers questions based ONLY on the document content
- Runs completely offline. No cloud. No API keys.

## Tech Stack

- **Gemma 4 E2B** (local LLM via Ollama)
- **Python** (orchestration)
- **PyPDF2** (PDF parsing)
- **ONNX** (lightweight local embeddings)
- **LanceDB** (local vector database)
- **Termux** (Android terminal environment)

## How to Run

1. Install Termux from F-Droid
2. Install Python and required packages
3. Pull Gemma 4 E2B via Ollama
4. Run the pipeline
5. Feed it a PDF and start asking questions

## Why This Matters

Built entirely from a phone. No laptop. No GPU. No cloud bills. Just open-source tools and a refusal to accept that AI development requires expensive hardware.

## Author

Okeke Chukwudubem
- GitHub: [Dexter2344](https://github.com/Dexter2344)
- Devto:  https://dev.to/okeke_chukwudubem_5f3bf49
