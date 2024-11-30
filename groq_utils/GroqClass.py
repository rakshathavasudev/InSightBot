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
        # os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY')
        # os.environ['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')

        self.pinecone_api_key = os.environ.get('PINECONE_API_KEY')
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        self.model_name = model_name

        # Initialize the Groq client with the provided API key
        self.groq_client = Groq(api_key=self.groq_api_key, http_client=None)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=self.pinecone_api_key)

        self.conversation_history=[]

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

        conversations="\n".join([f"{msg['role'].upper()}:{msg['content']}" for msg in self.conversation_history])
        augmented_query += f"Conversation History:\n{conversations}\n\nMy Question:\n{query}"
        # Define the system prompt for Groq
        system_prompt = '''
            You are an expert in reading the transcript of the given youtube video.
        Answer any question I have based on the transcripts i have provided. Mention the timestamps where the answer is given.
        Convert all the timestamps into minutes that are currently in seconds.
        '''

        # Make the call to Groq's chat completions
        res = self.groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile", # Specify the Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        bot_response= res.choices[0].message.content

        #Update conversations
        self.conversation_history.append({'role':'user','content':query})
        self.conversation_history.append({'role':'assistant','content':bot_response})

        return bot_response
    
    def reset_memory(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []
