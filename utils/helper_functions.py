import os

def delete_files_in_directory(directory_path, file_extension=None):
    """
    Deletes all files in the specified directory. If a file extension is provided,
    only files with that extension will be deleted.
    
    Args:
        directory_path (str): Path to the directory.
        file_extension (str, optional): Specific file extension to delete (e.g., ".txt").
    
    Returns:
        int: Number of files deleted.
    
    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")
    
    files_deleted = 0
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            if file_extension and not filename.endswith(file_extension):
                continue  # Skip files that don't match the specified extension
            
            os.remove(file_path)
            files_deleted += 1
    
    return files_deleted
