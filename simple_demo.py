import chromadb
from chromadb.config import Settings


client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory="content/"
                                ))

# collection = client.create_collection("user1")
collection = client.get_collection("user1")

# collection.add(
#     documents=["This is a document containing crypto project information",
#     "This is a document containing information about tokenization", 
#     "This document contains info about team"],
#     metadatas=[{"source": "Whitepaper"},{"source": "Tokenomics"},{'source':'Team'}],
#     ids=["id1", "id2", "id3"]
# )

collection.update(
    ids=["id2"],
    documents=["This is a document containing information about assets, characters, mechanics."],
    metadatas=[{"source": "Tokenomics"}],
)

results = collection.query(
    query_texts=["gaming"],
    n_results=2
)


print(results)