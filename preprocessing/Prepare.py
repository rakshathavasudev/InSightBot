import os
import time
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

class DocumentProcessor:
    def __init__(self, index_name: str, namespace: str, chunk_size: int = 1000, chunk_overlap: int = 30):
        self.index_name = index_name
        self.namespace = namespace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def delete_files_in_directory(self, directory_path: str):
        """
        Delete all files in a given directory.
        """
        files = os.listdir(directory_path)
        print('Files in dir:', files)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")

    def process_directory(self, directory_path: str):
        """
        Process all .txt files in a directory and load their content.
        """
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

    def prepare_data(self, documents):
        """
        Prepare the text content for embedding.
        """
        document_data = []
        for document in documents:
            document_source = document['Data'][0].metadata['source']
            document_content = document['Data'][0].page_content

            file_name = document_source.split("/")[-1]
            folder_names = document_source.split("/")[2:-1]

            doc = Document(
                page_content=f"<Source>\n{document_source}\n</Source>\n\n<Content>\n{document_content}\n</Content>",
                metadata={
                    "file_name": file_name,
                    "parent_folder": folder_names[-1],
                    "folder_names": folder_names
                }
            )
            document_data.append(doc)

        return document_data

    def chunk_data(self, docs):
        """
        Split documents into chunks based on the provided chunk size and overlap.
        """
        textsplitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = textsplitter.split_documents(docs)
        return docs

    def upsert_vectorstore_to_pinecone(self, document_data, embeddings):
        """
        Upsert data into Pinecone vector store after handling namespace.
        """
        # Initialize Pinecone connection
        pc = pinecone.Pinecone(api_key=os.environ['PINECONE_API_KEY'])

        # Check if the namespace exists in the index
        index = pc.Index(self.index_name)
        namespaces = index.describe_index_stats().get('namespaces', [])
        
        max_retries = 5
        wait_time = 2  # In seconds
        if self.namespace in namespaces:
            print(f"Namespace '{self.namespace}' found. Deleting vector data...")
            index.delete(namespace=self.namespace, delete_all=True)

            # Polling to ensure deletion completes
            for attempt in range(max_retries):
                namespaces = index.describe_index_stats().get('namespaces', [])
                if self.namespace not in namespaces:
                    print(f"Namespace '{self.namespace}' deletion confirmed.")
                    break
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to delete namespace '{self.namespace}' after {max_retries} retries.")
        else:
            print(f"Namespace '{self.namespace}' does not exist. Proceeding with upsert.")

        # Create or replace the vector store
        vectorstore_from_documents = PineconeVectorStore.from_documents(
            document_data,
            embeddings,
            index_name=self.index_name,
            namespace=self.namespace
        )
        print(f"Vector store type: {type(vectorstore_from_documents)}")
        return vectorstore_from_documents

    def initialize_pinecone(self):
        """
        Initialize Pinecone with the provided index name.
        """
        # Initialize Pinecone instance
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

        # Check if the index exists
        if self.index_name not in [index.name for index in pc.list_indexes().indexes]:
            raise ValueError(f"Index '{self.index_name}' does not exist. Please create it first.")

        # Connect to the specified index
        pinecone_index = pc.Index(self.index_name)
        return pinecone_index
