import os
import streamlit as st
from preprocessing.file_processor import process_file
from embedding.embedding_utils import get_huggingface_embeddings
from embedding.pinecone_utils import upload_to_pinecone
from utils.helper_functions import delete_files_in_directory

# Configure Streamlit page
st.set_page_config(page_title="Embedding Manager", layout="wide")

# Directory paths
UPLOAD_DIR = "uploaded_files"
PROCESSED_DIR = "processed_files"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Streamlit App
def main():
    st.title("Embedding Manager")

    # Sidebar options
    with st.sidebar:
        st.header("Navigation")
        selected_option = st.radio(
            "Choose an action:",
            options=["Upload and Process Files", "View Embeddings", "Manage Files"]
        )

    # Option 1: Upload and Process Files
    if selected_option == "Upload and Process Files":
        st.header("Upload and Process Files")

        uploaded_files = st.file_uploader(
            "Upload files (text only):",
            type=["txt", "csv"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"Uploaded {uploaded_file.name} successfully!")

            if st.button("Process Uploaded Files"):
                for filename in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    processed_path = os.path.join(PROCESSED_DIR, filename)

                    # Process the file
                    try:
                        processed_text = process_file(file_path)
                        with open(processed_path, "w") as f:
                            f.write(processed_text)
                        st.success(f"Processed {filename} successfully!")
                    except Exception as e:
                        st.error(f"Error processing {filename}: {e}")

    # Option 2: View Embeddings
    elif selected_option == "View Embeddings":
        st.header("Generate and View Embeddings")

        files = os.listdir(PROCESSED_DIR)
        if files:
            selected_file = st.selectbox("Select a file to generate embeddings:", files)

            if selected_file:
                file_path = os.path.join(PROCESSED_DIR, selected_file)
                with open(file_path, "r") as f:
                    text = f.read()

                if st.button("Generate Embeddings"):
                    try:
                        embeddings = get_huggingface_embeddings(text)
                        st.success("Embeddings generated successfully!")
                        st.json(embeddings)

                        if st.button("Upload to Pinecone"):
                            upload_to_pinecone(embeddings, metadata={"filename": selected_file})
                            st.success("Uploaded embeddings to Pinecone!")
                    except Exception as e:
                        st.error(f"Error generating embeddings: {e}")
        else:
            st.warning("No processed files available. Please upload and process files first.")

    # Option 3: Manage Files
    elif selected_option == "Manage Files":
        st.header("Manage Files")

        st.subheader("Uploaded Files")
        uploaded_files = os.listdir(UPLOAD_DIR)
        st.write(uploaded_files)

        if st.button("Clear Uploaded Files"):
            deleted_count = delete_files_in_directory(UPLOAD_DIR)
            st.success(f"Deleted {deleted_count} uploaded files.")

        st.subheader("Processed Files")
        processed_files = os.listdir(PROCESSED_DIR)
        st.write(processed_files)

        if st.button("Clear Processed Files"):
            deleted_count = delete_files_in_directory(PROCESSED_DIR)
            st.success(f"Deleted {deleted_count} processed files.")

if __name__ == "__main__":
    main()
