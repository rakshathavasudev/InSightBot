from flask import Flask, render_template, request, jsonify
from groq_utils import GroqClass
from preprocessing import DocumentProcessor
from transcriber import YouTubeTranscriber
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
app = Flask(__name__)
# Load environment variables from .env file
load_dotenv()

index_name = "insightbot"
namespace = "transcripts"
# Set environment variables from .env
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Initialize embeddings and shared classes
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
groq_client = GroqClass()  # Initialize GroqClass globally to be reused
document_processor = DocumentProcessor(index_name=index_name,namespace=namespace)  # Initialize DocumentProcessor globpally
# GLOBAL
document_dir_path = "resources/documents"
transcript_dir_path = "resources/transcripts"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/submit_media', methods=['POST'])
def submit_media():
    try:
        document_processor.delete_files_in_directory(transcript_dir_path)
        document_processor.delete_files_in_directory(document_dir_path)

        # Get YouTube links and documents from the request
        youtube_links = request.form.getlist('youtube_links[]')  # List of YouTube links
        files = request.files.getlist('documents[]')  # List of uploaded files

        # Validate input
        if not youtube_links and not files:
            return jsonify({"error": "At least one YouTube link or document is required"}), 400

        all_documents = []

        # Process each YouTube link
        if youtube_links:
            print("=== Processing YouTube Links ===")
            for youtube_link in youtube_links:
                if youtube_link:
                    print(f"Processing YouTube Link: {youtube_link}")
                    transcriber = YouTubeTranscriber(youtube_link, output_dir = transcript_dir_path)
                    transcriber.transcribe()
                    print(f"Transcription complete for: {youtube_link}")
            documents = document_processor.process_directory(transcript_dir_path)
            all_documents.extend(documents)

        # Process uploaded documents
        if files:
            print("=== Processing Uploaded Documents ===")
            for file in files:
                file_path = os.path.join(document_dir_path, file.filename)
                file.save(file_path)
                print(f"Saved file: {file_path}")
            uploaded_documents = document_processor.process_directory(document_dir_path)
            all_documents.extend(uploaded_documents)

        if not all_documents:
            return jsonify({"error": "No valid documents to process"}), 400

        # Prepare and chunk data
        document_data = document_processor.prepare_data(all_documents)
        documents_chunks = document_processor.chunk_data(document_data)

        # Upsert to Pinecone or your vector store
        index_name = "insight-bot"
        namespace = "media-data"
        vectorstore_from_documents = document_processor.upsert_vectorstore_to_pinecone(documents_chunks, embeddings, index_name, namespace)
        print('=== Upsert to Pinecone done ===', vectorstore_from_documents)

        return jsonify({"message": "Media processed successfully!"})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Failed to process media", "details": str(e)}), 500

@app.route('/submit_youtube_link', methods=['POST'])

# def submit_youtube_link():
#     directory_path = "/transcripts"
#     youtube_link = request.form.get('youtube_link')

#     if not youtube_link:
#         return jsonify({"error": "No YouTube link provided"}), 400

#     # Process documents
#     try:
#         document_processor.delete_files_in_directory(directory_path)
#         transcriber = YouTubeTranscriber(youtube_link)
#         transcriber.transcribe()

#         documents = document_processor.process_directory(directory_path)
#         document_data = document_processor.prepare_data(documents)
#         documents_chunks = document_processor.chunk_data(document_data)

#         index_name = "insightbot"
#         namespace = "transcripts"

#         # Upsert vectorstore to Pinecone
#         vectorstore_from_documents = document_processor.upsert_vectorstore_to_pinecone(
#             documents_chunks, embeddings, index_name, namespace
#         )
        
#         return jsonify({"message": f"Link '{youtube_link}' submitted and processed successfully!"})

#     except Exception as e:
#         return jsonify({"error": f"Error processing YouTube link: {str(e)}"}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    # Get the response from the model
    pinecone_index = groq_client.initialize_pinecone(index_name)
    answer=groq_client.perform_rag(pinecone_index, namespace, question)
    return jsonify({"question": question, "answer": answer})

# Simulating model response for now
def get_model_response(question):
    # Replace with actual logic to handle YouTube video and query
    return f"Answer to your question '{question}'"


if __name__ == "__main__":
    app.run(debug=True)
