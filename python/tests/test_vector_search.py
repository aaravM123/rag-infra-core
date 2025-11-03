import sys
import os
import numpy as np
import time

# Add parent directory (python/) to path to import vector_search wrapper
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import vector_search  # type: ignore

a = np.random.rand(1536).astype(np.float32)
b = np.random.rand(1536).astype(np.float32)

t0 = time.time()
cpp_cos = vector_search.cosine_similarity(a, b)
t1 = time.time()
py_cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
t2 = time.time()

print(f"C++ cosine ={cpp_cos}, Python cosine ={py_cos:.6f}")
print(f"Speed: C++={t1-t0:.6f}s, Python={t2-t1:.6f}s")

vecs = [np.random.rand(1536).astype(np.float32).tolist() for i in range(5000)]
query = np.random.rand(1536).astype(np.float32).tolist()
print("Top-5 indices:", vector_search.search_topk(vecs, query, 5))