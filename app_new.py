from flask import Flask, render_template, request, jsonify
from .groq_utils import GroqClass
from .preprocessing import DocumentProcessor
from .transcriber import YouTubeTranscriber
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

# Initialize embeddings and shared classes
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
groq_client = GroqClass()  # Initialize GroqClass globally to be reused
document_processor = DocumentProcessor()  # Initialize DocumentProcessor globally

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit_youtube_link', methods=['POST'])
def submit_youtube_link():
    directory_path = "/transcripts"
    youtube_link = request.form.get('youtube_link')

    if not youtube_link:
        return jsonify({"error": "No YouTube link provided"}), 400

    # Process documents
    try:
        document_processor.delete_files_in_directory(directory_path)
        transcriber = YouTubeTranscriber(youtube_link)
        transcriber.transcribe()

        documents = document_processor.process_directory(directory_path)
        document_data = document_processor.prepare_data(documents)
        documents_chunks = document_processor.chunk_data(document_data)

        index_name = "insightbot"
        namespace = "transcripts"

        # Upsert vectorstore to Pinecone
        vectorstore_from_documents = document_processor.upsert_vectorstore_to_pinecone(
            documents_chunks, embeddings, index_name, namespace
        )
        
        return jsonify({"message": f"Link '{youtube_link}' submitted and processed successfully!"})

    except Exception as e:
        return jsonify({"error": f"Error processing YouTube link: {str(e)}"}), 500


@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get('question')

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Answering question using the initialized GroqClass
    try:
        index_name = "insightbot"
        namespace = "transcripts"

        # Initialize Pinecone and get answer
        pinecone_index = groq_client.initialize_pinecone(index_name)
        answer = groq_client.perform_rag(pinecone_index, namespace, question)

        return jsonify({"question": question, "answer": answer})

    except Exception as e:
        return jsonify({"error": f"Error answering the question: {str(e)}"}), 500


# Simulating model response for now
def get_model_response(question):
    # Replace with actual logic to handle YouTube video and query
    return f"Answer to your question '{question}'"


if __name__ == "__main__":
    app.run(debug=True)
