from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

class GroqClass:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the class with necessary keys and model for embeddings.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Set environment variables from .env
        os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
        os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

        self.pinecone_api_key = os.environ['PINECONE_API_KEY']
        self.groq_api_key = os.environ['GROQ_API_KEY']
        self.model_name = model_name

        # Initialize the Groq client with the provided API key
        self.groq_client = Groq(api_key=self.groq_api_key, http_client=None)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)

    def get_huggingface_embeddings(self, text):
        """
        Get the Hugging Face embeddings for a given text.
        """
        model = SentenceTransformer(self.model_name)
        return model.encode(text)
    
    def initialize_pinecone(self, index_name: str):
        """
        Initialize Pinecone connection and verify if the index exists.
        """
        # Check if the index already exists; if not, raise an error or handle accordingly
        if index_name not in [index.name for index in self.pinecone.list_indexes().indexes]:
            raise ValueError(f"Index '{index_name}' does not exist. Please create it first.")

        # Connect to the specified index
        pinecone_index = self.pinecone.Index(index_name)

        return pinecone_index

    def perform_rag(self, pinecone_index, namespace, query):
        """
        Perform Retrieval-Augmented Generation (RAG) by querying Pinecone and using Groq for response.
        """
        # Get the embeddings for the query
        raw_query_embedding = self.get_huggingface_embeddings(query)

        query_embedding = np.array(raw_query_embedding)

        # Query Pinecone to retrieve top matches
        top_matches = pinecone_index.query(vector=query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)

        # Extract contexts from the matches
        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        # Augment the query with the context information
        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

        # Define the system prompt for Groq
        system_prompt = '''
            You are a skilled expert in analyzing and understanding textual content from various sources, including YouTube video transcripts and document files.
            Your task is to answer any questions I have based on the provided text.
            If timestamps are present in seconds, convert them into a minutes:seconds format (e.g., 90 seconds becomes 1:30).
            Respond clearly and concisely with complete accuracy.
        '''

        # Make the call to Groq's chat completions
        res = self.groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile", # Specify the Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        return res.choices[0].message.content
