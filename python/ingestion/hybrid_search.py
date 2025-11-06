import numpy as np
import vector_search
from ingestion.db_client import fetch_embeddings

def hybrid_search(query_emb: np.ndarray, k_pg=10, k_cpp=5):
    # Retrieve candidate IDs and Embeddings from PostgreSQL
    ids, vectors = fetch_embeddings(limit=5000)

    top_cpp = vector_search.search_topk_np(vectors, query_emb.astype(np.float32), k_cpp)

    top_ids = [int(ids[i]) for i in top_cpp]

    return top_ids