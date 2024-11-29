import pinecone

def initialize_pinecone(api_key, environment):
    """
    Initialize Pinecone with the provided API key and environment.
    
    Args:
        api_key (str): Pinecone API key.
        environment (str): Pinecone environment.
    """
    pinecone.init(api_key=api_key, environment=environment)

def upsert_embeddings(index_name, embeddings, metadata_list):
    """
    Upsert embeddings into a Pinecone index.
    
    Args:
        index_name (str): Pinecone index name.
        embeddings (list of list of float): Embedding vectors to upsert.
        metadata_list (list of dict): Metadata for each embedding.
    """
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} does not exist.")
    
    index = pinecone.Index(index_name)
    
    # Prepare the data for upsert
    vectors = [
        {
            "id": f"doc-{i}",
            "values": embedding,
            "metadata": metadata
        }
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list))
    ]
    index.upsert(vectors)

def query_pinecone(index_name, query_vector, top_k=5):
    """
    Query a Pinecone index to find the top-k most similar vectors.
    
    Args:
        index_name (str): Pinecone index name.
        query_vector (list of float): Query vector.
        top_k (int): Number of results to retrieve.
    
    Returns:
        list of dict: Query results with metadata and similarity scores.
    """
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index {index_name} does not exist.")
    
    index = pinecone.Index(index_name)
    
    # Query the index
    results = index.query(query_vector, top_k=top_k, include_metadata=True)
    return results["matches"]
