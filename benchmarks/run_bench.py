import numpy as np, time, json, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import vector_search  # type: ignore


def benchmark(num_vectors=10000, dim = 1536, k=5):
    vecs = np.random.rand(num_vectors, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)

    # Python
    t0 = time.time()
    py_scores = vecs @ query / (
         np.linalg.norm(vecs, axis=1) * np.linalg.norm(query)
    )

    py_topk = np.argsort(py_scores)[::-1][:k]
    t1 = time.time()

    # C++
    vec_list = [v.tolist() for v in vecs]
    query_list = query.tolist()
    t2 = time.time()
    cpp_topk = vector_search.search_topk(vec_list, query_list, k)
    t3 = time.time()

    return {
        "vectors": num_vectors,
        "dim": dim,
        "k": k,
        "python_time": round(t1-t0, 4),
        "cpp_time": round(t3-t2, 4),
        "speedup": round(t0-t1 / (t3-t2), 4),
        "topk_match": len(set(py_topk).intersection(cpp_topk)),
    }

if __name__ == "__main__":
    results = []
    for n in [1000, 5000, 10000, 20000]:
        print(f"Running benchmark with {n} vectors...")
        res = benchmark(num_vectors=n)
        print(res)
        results.append(res)

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark results saved to benchmarks/results.json")