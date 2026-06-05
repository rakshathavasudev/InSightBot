import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
from groq_utils import GroqClass
from preprocessing import DocumentProcessor
from transcriber import YouTubeTranscriber
from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)

index_name = "insightbot"
namespace = "transcripts"

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
        groq_client.reset_memory()
        document_processor.delete_files_in_directory(transcript_dir_path)
        document_processor.delete_files_in_directory(document_dir_path)

        # Get YouTube links and documents from the request
        youtube_links = request.form.getlist('youtube_links[]')  # List of YouTube links
        files = request.files.getlist('documents[]')  # List of uploaded files

        # Validate input
        if not youtube_links and not files:
            return jsonify({"error": "At least one YouTube link or document is required"}), 400

        all_documents = []
        failed_youtube = []
        failed_uploaded = []

        # Process each YouTube link
        if youtube_links:
            print("=== Processing YouTube Links ===")
            import re
            for youtube_link in youtube_links:
                if youtube_link:
                    # Normalize the link: extract the 11-char video id and build a canonical youtu.be URL
                    vid = None
                    # common patterns: watch?v=VIDEOID or youtu.be/VIDEOID
                    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', youtube_link)
                    if m:
                        vid = m.group(1)
                        clean_link = f'https://youtu.be/{vid}'
                    else:
                        # fallback: pass original link but log for debugging
                        clean_link = youtube_link
                    print(f"Processing YouTube Link: {clean_link}")
                    transcriber = YouTubeTranscriber(clean_link, output_dir = transcript_dir_path)
                    transcriber.transcribe()
                    # After attempting transcription, check whether a transcript file was created (single-video case)
                    try:
                        vid = getattr(transcriber, 'video_id', None)
                        if vid:
                            expected_path = os.path.join(transcript_dir_path, f"{vid}_transcript.txt")
                            if not os.path.exists(expected_path):
                                print(f"No transcript file created for video ID {vid}")
                                failed_youtube.append(youtube_link)
                            else:
                                print(f"Transcription complete for: {youtube_link}")
                        else:
                            # Playlist or unknown - we'll rely on directory processing to find any transcripts
                            print(f"Transcription attempted for (playlist or unknown id): {youtube_link}")
                    except Exception as _:
                        # Non-fatal — record as failure to help debugging
                        failed_youtube.append(youtube_link)
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
            # Record which uploaded files failed to produce content
            uploaded_filenames = [f.filename for f in files]
            processed_filenames = [os.path.basename(d['File']) for d in uploaded_documents]
            for fname in uploaded_filenames:
                if fname not in processed_filenames:
                    failed_uploaded.append(fname)
            all_documents.extend(uploaded_documents)

        if not all_documents:
            # Provide helpful debugging details about failures
            details = {
                "error": "No valid documents to process",
                "failed_youtube_links": failed_youtube,
                "failed_uploaded_files": failed_uploaded,
                "note": "YouTube videos may not have transcripts available. PDFs that are scanned images need OCR to extract text."
            }
            return jsonify(details), 400

        # Prepare and chunk data
        document_data = document_processor.prepare_data(all_documents)
        documents_chunks = document_processor.chunk_data(document_data)

        vectorstore_from_documents = document_processor.upsert_vectorstore_to_pinecone(documents_chunks, embeddings, index_name, namespace)
        print('=== Upsert to Pinecone done ===', vectorstore_from_documents)

        return jsonify({"message": "Media processed successfully!"})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Failed to process media", "details": str(e)}), 500

@app.route('/submit_youtube_link', methods=['POST'])

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


# if __name__ == "__main__":
#     app.run(debug=False)
if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8080)))