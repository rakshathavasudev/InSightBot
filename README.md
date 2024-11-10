# InSightBot

## About
InSightBot is a Retrieval-Augmented Generation (RAG) based chatbot that allows users to input youtube video links and ask questions about the content, receiving contextually relevant answers. The bot leverages video transcripts, extracts meaningful data, and uses a combination of retrieval and generation techniques to respond to user queries accurately.

## System design 

![image](https://github.com/user-attachments/assets/4bfae624-e2b0-413f-8cb3-ecb795f84db3)

## Architecture
InSightBot uses a multi-layered architecture to process youtube video transcripts and generate responses. Key components include:

Data Ingestion/Processing Layer: Extracts and preprocesses video transcripts. 

Storage & Indexing Layer: Stores transcript data and embeds it in a vector database for efficient retrieval.

RAG and Retrieval layer: Handles user questions, retrieves relevant transcript sections, and uses a language model to provide answers.

## Workflow
Transcript Extraction: Ingests YouTube links  and retrieves transcripts in files.

Chunking & Embedding: Splits transcripts into chunks, vectorizes them, and stores them in a vector database(Picone).

RAG Process: When a user asks a question, the bot retrieves relevant chunks and generates a response using a language model.


