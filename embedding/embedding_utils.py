from transformers import AutoTokenizer, AutoModel
import torch

def get_huggingface_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings for a list of texts using a Hugging Face model.
    
    Args:
        texts (list of str): List of texts to embed.
        model_name (str): Hugging Face model name.
    
    Returns:
        list of list of float: Generated embeddings.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings = []
    for text in texts:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean pooling strategy
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        embeddings.append(embedding)
    
    return embeddings
