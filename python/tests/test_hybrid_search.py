import numpy as np
from ingestion.hybrid_search import hybrid_search

query = np.random.rand(1536).astype(np.float32)
results = hybrid_search(query)
print("Top 5 results:", results)