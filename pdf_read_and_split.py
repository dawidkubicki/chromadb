from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings


# Loading document and splitting into the chunks
pdfLoader = PyMuPDFLoader('10k/apple.pdf')
documents = pdfLoader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Loading Embeddings
modelPath = "all-MiniLM-L6-v2" # or sentence-transformers/all-mpnet-base-v2
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name = modelPath,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
)


# # Store vector data (embeddings)
# db = FAISS.from_documents(docs, embeddings)
# question = "What are he legal proceedings particularly in 2020?"
# searchDocs = db.similarity_search(question)
# # print(searchDocs[0].page_content)