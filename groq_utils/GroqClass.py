from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

class GroqClass:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        load_dotenv()
        self.model_name = model_name
        self.groq_client = None
        self.embeddings = None
        self.pinecone = None
        self.conversation_history = []

    def load_groq_client(self):
        if self.groq_client is None:
            self.groq_client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    def load_embeddings(self):
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def load_pinecone(self):
        if self.pinecone is None:
            self.pinecone = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

    def get_huggingface_embeddings(self, text):
        model = SentenceTransformer(self.model_name)  # keep this here to load on demand
        return model.encode(text)
    
    def initialize_pinecone(self, index_name: str):
        self.load_pinecone()

        if index_name not in [index.name for index in self.pinecone.list_indexes().indexes]:
            raise ValueError(f"Index '{index_name}' does not exist.")

        return self.pinecone.Index(index_name)

    def perform_rag(self, pinecone_index, namespace, query):
        self.load_groq_client()

        query_embedding = np.array(self.get_huggingface_embeddings(query))

        top_matches = pinecone_index.query(vector=query_embedding.tolist(), top_k=10, include_metadata=True, namespace=namespace)
        contexts = [item['metadata']['text'] for item in top_matches['matches']]

        augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query
        conversations = "\n".join([f"{msg['role'].upper()}:{msg['content']}" for msg in self.conversation_history])
        augmented_query += f"\nConversation History:\n{conversations}\n\nMy Question:\n{query}"

        system_prompt = '''
            You are an expert in reading the transcript of the given YouTube video.
            Answer based on the transcripts. Mention timestamps in minutes (not seconds).
        '''

        res = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ]
        )

        bot_response = res.choices[0].message.content
        self.conversation_history.append({'role': 'user', 'content': query})
        self.conversation_history.append({'role': 'assistant', 'content': bot_response})

        return bot_response

    def reset_memory(self):
        self.conversation_history = []
