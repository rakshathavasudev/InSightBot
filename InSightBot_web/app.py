from flask import Flask, render_template, request, jsonify
import os
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import os
from groq import Groq
import pinecone
import time
from dotenv import load_dotenv

from transcriber import YouTubeTranscriber

load_dotenv()
app = Flask(__name__)
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

pinecone_api_key= '53f1d7ac-54ea-48e4-b4d5-44b410c3fd7b'
groq_api_key = "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"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

class YouTubeTranscriber:
    def __init__(self, url, transcript_language='en', output_dir='transcripts'):
        """
        Initializes the YouTubeTranscriber with a URL (playlist or single video), transcript language, and output directory.

        :param url: URL of the YouTube playlist or single video
        :param transcript_language: Preferred language for transcripts (default is 'en' for English)
        :param output_dir: Directory where transcripts will be saved
        """
        self.url = url
        self.transcript_language = transcript_language
        self.output_dir = output_dir

        # Check if the URL is for a playlist or a single video
        if 'playlist?list=' in url:
            self.is_playlist = True
            self.playlist = Playlist(url)
            print('self.playlist', self.playlist)
        elif 'watch?v=' or 'youtu.be/' in url:
            self.is_playlist = False
            self.video_id = url.split('=')[-1]
        else:
            raise ValueError("Invalid URL. Provide a valid YouTube playlist or video URL.")

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        if self.is_playlist:
            print(f"Found {len(self.playlist.video_urls)} videos in the playlist.")

    def fetch_transcript(self, video_id):
        """
        Fetches the transcript for a single video in the specified language.

        :param video_id: YouTube video ID
        :return: Transcript as a list of dictionaries with 'start' and 'text' keys
        """
        try:
            # Fetch transcript with the specified language
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[self.transcript_language])
            return transcript
        except Exception as e:
            print(f"Could not retrieve transcript for video ID {video_id}: {e}")
            return None

    def save_transcript_to_file(self, video_id, transcript):
        """
        Saves the transcript of a video to a text file.

        :param video_id: YouTube video ID
        :param transcript: Transcript data to save
        """
        file_path = os.path.join(self.output_dir, f"{video_id}_transcript.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            for line in transcript:
                file.write(f"{line['start']}: {line['text']}\n")
        print(f"Transcript saved for video ID: {video_id}")

    def transcribe_playlist(self):
        """
        Processes each video in the playlist to fetch and save transcripts.
        """
        for video_url in self.playlist.video_urls:
            # Extract video ID from URL
            video_id = video_url.split('=')[-1]
            # Fetch and save the transcript
            transcript = self.fetch_transcript(video_id)
            if transcript:
                self.save_transcript_to_file(video_id, transcript)

    def transcribe_single_video(self):
        """
        Fetches and saves the transcript for a single YouTube video.
        """
        # Fetch and save the transcript
        transcript = self.fetch_transcript(self.video_id)
        if transcript:
            self.save_transcript_to_file(self.video_id, transcript)

    def transcribe(self):
        """
        Determines if the URL is a playlist or single video and processes accordingly.
        """
        if self.is_playlist:
            self.transcribe_playlist()
        else:
            self.transcribe_single_video()

#==========================#

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def process_directory(directory_path):
    data = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):  # Only process .txt files
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                # Use TextLoader to load the content
                loader = TextLoader(file_path)
                data.append({"File": file_path, "Data": loader.load()})

    return data

def prepare_data(documents):
  # Prepare the text for embedding
  document_data = []
  for document in documents:

      document_source = document['Data'][0].metadata['source']
      document_content = document['Data'][0].page_content

      file_name = document_source.split("/")[-1]
      folder_names = document_source.split("/")[2:-1]

      doc = Document(
          page_content = f"<Source>\n{document_source}\n</Source>\n\n<Content>\n{document_content}\n</Content>",
          metadata = {
              "file_name": file_name,
              "parent_folder": folder_names[-1],
              "folder_names": folder_names
          }
      )
      document_data.append(doc)

  return document_data


def chunk_data(docs, chunk_size=1000,chunk_overlap=30):
    textsplitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs=textsplitter.split_documents(docs)
    return docs

def upsert_vectorstore_to_pinecone(document_data, embeddings, index_name, namespace):
    # # Initialize Pinecone connection with the new API structure
    pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])

    # Check if the namespace exists in the index
    index = pc.Index(index_name)

    # Check if the namespace exists by listing the namespaces (or by trying to query)
    namespaces = index.describe_index_stats().get('namespaces', [])
    max_retries = 5
    wait_time = 2000
    if namespace in namespaces:
        print(f"Namespace '{namespace}' found. Deleting vector data...")
        index.delete(namespace=namespace, delete_all=True)  # Initiates deletion

        # Polling to ensure deletion completes
        for attempt in range(max_retries):
            namespaces = index.describe_index_stats().get('namespaces', [])
            if namespace not in namespaces:
                print(f"Namespace '{namespace}' deletion confirmed.")
                break
            time.sleep(wait_time)  # Wait before re-checking
        else:
            raise RuntimeError(f"Failed to delete namespace '{namespace}' after {max_retries} retries.")

    else:
        print(f"Namespace '{namespace}' does not exist. Proceeding with upsert.")
    
    # Create or replace the vector store
    vectorstore_from_documents = PineconeVectorStore.from_documents(
        document_data,
        embeddings,
        index_name=index_name,
        namespace=namespace
    )
    print(f"Vector store type: {type(vectorstore_from_documents)}")
    # Optionally, return the vector store if needed
    return vectorstore_from_documents

def initialize_pinecone(index_name: str):

  # Create an instance of the Pinecone class

  pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

  #Check if the index already exists; if not, raise an error or handle accordingly
  if index_name not in [index.name for index in pc.list_indexes().indexes]:
    raise ValueError(f"Index rais '{index_name}' does not exist. Please create it first.")

  # Connect to the specified index
  pinecone_index = pc. Index(index_name)

  return pinecone_index


def perform_rag(pinecone_index, namespace, query):
    raw_query_embedding = get_huggingface_embeddings(query)

    query_embedding = np.array(raw_query_embedding)

    top_matches = pinecone_index.query(vector=query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
    print('augmented_query', augmented_query)
    # Modify the prompt below as need to improve the response quality
    system_prompt = f'''
    You are an expert in reading the transcript of the given youtube video.
    Answer any question I have based on the transcripts i have provided. Mention the timestamps where the answer is given.
    Convert all the timestamps into minutes that are currently in seconds.
    '''

    res = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile", # llama-3.1-70b-versatile
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return res.choices[0].message.content

def delete_files_in_directory(directory_path):
    files = os.listdir(directory_path)
    print('files in dir', files)
    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("All files deleted successfully.")


# ========= FLASK ========= #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit_youtube_link', methods=['POST'])
def submit_youtube_link():
    directory_path = "/transcripts"
    youtube_link = request.form.get('youtube_link')
    delete_files_in_directory('transcripts')
    print("=== delete_files_in_directory")
    print("=== got youtube_link", youtube_link)
    transcriber = YouTubeTranscriber(youtube_link)
    transcriber.transcribe()
    print('===transcriber done')
    documents = process_directory(directory_path)
    document_data=prepare_data(documents)
    documents_chunks=chunk_data(document_data)
    
    index_name = "insightbot"
    namespace = "transcripts"
    # pinecone_index = initialize_pinecone(index_name)
    # print('pinecone_index', pinecone_index)

    vectorstore_from_documents = upsert_vectorstore_to_pinecone(documents_chunks, embeddings, index_name, namespace)
    print('===upsert_vectorstore_to_pinecone done', vectorstore_from_documents)
    
    # Process the YouTube link (you can add logic to extract video details)
    return jsonify({"message": f"Link '{youtube_link}' submitted successfully!"})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    # Get the response from the model
    index_name = "insightbot"
    namespace = "transcripts"
    pinecone_index = initialize_pinecone(index_name)
    answer=perform_rag(pinecone_index, namespace, question)
    return jsonify({"question": question, "answer": answer})

# Simulating model response for now
def get_model_response(question):
    # Replace with actual logic to handle YouTube video and query
    return f"Answer to your question '{question}'"

if __name__ == "__main__":
    app.run(debug=True)
