import shutil, os
from pathlib import Path

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma


directory = './raw_documents/'
user_bucket_name = 'user1_bucket'

def load_docs(directory: str):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def collection_exists(user_bucket_name: str, base_directory: str = "chroma_embeddings/") -> bool:
    """
    Check if a ChromaDB collection exists for a given user.

    Args:
    user_bucket_name (str): Name of the user bucket.
    base_directory (str): Base directory where ChromaDB collections are stored.

    Returns:
    bool: True if collection exists, False otherwise.
    """
    collection_path = f"{base_directory}{user_bucket_name}"
    return os.path.exists(collection_path)

def create_collection(user_folder_name: str, model_name: str, s3_base_directory: str = "s3_documents/", chroma_base_directory: str = "chroma_embeddings/"):
    """
    Create a new ChromaDB collection for a user, where each document's embeddings are stored in a separate folder.

    Args:
    user_folder_name (str): Name of the user's folder containing raw documents.
    model_name (str): Name of the model for embedding generation.
    s3_base_directory (str): Base directory where users' raw documents are stored.
    chroma_base_directory (str): Base directory where ChromaDB collections are stored.
    """
    user_s3_path = Path(s3_base_directory) / user_folder_name
    user_chroma_path = Path(chroma_base_directory) / user_folder_name

    # Check if the user's raw documents directory exists
    if not user_s3_path.exists():
        print(f"No documents found in {user_s3_path}.")
        return

    # Create directory for user's ChromaDB collection if it doesn't exist
    if not user_chroma_path.exists():
        os.makedirs(user_chroma_path)

    # Initialize the embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    # document_paths = load_docs(str(user_s3_path))
    for doc_file in user_s3_path.iterdir():
        loaded_doc = load_docs(str(doc_file))
        print(loaded_doc)
        break

        # if doc_file.is_file():
        #     doc_name = doc_file.stem  # Gets the file name without extension
        #     doc_embedding_path = user_chroma_path / doc_name
        #     os.makedirs(doc_embedding_path, exist_ok=True)


    

    # # Process and store embeddings for each document
    # for doc_path in document_paths:
    #     doc_name = Path(doc_path).stem
    #     doc_embedding_path = user_chroma_path / doc_name
    #     os.makedirs(doc_embedding_path, exist_ok=True)

    #     with open(doc_path, 'r') as file:
    #         doc_content = file.read()

    #     Chroma.from_documents([doc_content], embedding_function, persist_directory=str(doc_embedding_path))

    # print(f"Collection for {user_folder_name} has been created in {user_chroma_path}.")


def update_collection(new_docs, user_bucket_name: str, model_name: str, base_directory: str = "chroma_embeddings/"):
    """
    Update an existing ChromaDB collection with new documents, storing each in a separate folder.

    Args:
    new_docs (list): New documents to be added to the collection.
    user_bucket_name (str): Name of the user bucket.
    model_name (str): Name of the model for embedding generation.
    base_directory (str): Base directory where ChromaDB collections are stored.
    """
    collection_base_path = f"{base_directory}{user_bucket_name}/"

    if not collection_exists(user_bucket_name, base_directory):
        print(f"No existing collection found for {user_bucket_name}. Consider creating a new collection instead.")
        return

    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    existing_docs_count = len(os.listdir(collection_base_path))

    for idx, doc in enumerate(new_docs, start=existing_docs_count):
        doc_directory = f"{collection_base_path}doc_{idx}/"
        os.makedirs(doc_directory, exist_ok=True)
        Chroma.from_documents([doc], embedding_function, persist_directory=doc_directory)

    print(f"Collection for {user_bucket_name} has been updated with new documents.")

def delete_collection(user_bucket_name: str, base_directory: str = "chroma_embeddings/"):
    """
    Delete a ChromaDB collection for a given user.

    Args:
    user_bucket_name (str): Name of the user bucket.
    base_directory (str): Base directory where ChromaDB collections are stored.
    """
    collection_path = f"{base_directory}{user_bucket_name}"
    if collection_exists(user_bucket_name, base_directory):
        shutil.rmtree(collection_path)
        print(f"Collection for {user_bucket_name} has been deleted.")
    else:
        print(f"No collection found for {user_bucket_name}.")

def list_collections(base_directory: str = "chroma_embeddings/"):
    """
    List all ChromaDB collections in the base directory.

    Args:
    base_directory (str): Base directory where ChromaDB collections are stored.

    Returns:
    list: List of all collection names.
    """
    return [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]


# documents = load_docs(directory, user_bucket_name)
# print(len(documents))

# docs = split_docs(documents)
# print(len(docs))

# modelPath = "all-MiniLM-L6-v2" # or sentence-transformers/all-mpnet-base-v2
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")



# query = "revenue of nvidia in 2023"
# matching_docs = db.similarity_search(query,4)

# print(matching_docs)


### Testing use cases ====================================================================
# List all collections of users 
print(f"Listing all collections of users ...\n")
print(list_collections())

# Create a new collections for existing documents from a particular user bucket
print(f"Creating a new collections for existing documents from a particular user bucket ... \n")
create_collection(
    user_folder_name="user1_bucket",
    model_name="all-MiniLM-L6-v2",

)