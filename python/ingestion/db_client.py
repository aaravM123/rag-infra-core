import psycopg2
import numpy as np
import os
DB_CONFIG = dict(
    dbname = "ragdb",
    user = "postgres",
    password = "postgres",
    host="localhost",
    port="5433",
)

def fetch_embeddings(limit=1000):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT id, embedding FROM chunks LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
 

    ids, embs = zip(*rows)

    def _normalize_embedding(val):
        if isinstance(val, np.ndarray):
            return val.astype(np.float32)
        if isinstance(val, (list, tuple)):
            return np.asarray(val, dtype=np.float32)
        if isinstance(val, (bytes, bytearray, memoryview)):
            return np.frombuffer(val, dtype=np.float32)
        if isinstance(val, str):
            stripped = val.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                stripped = stripped[1:-1]
            return np.fromstring(stripped, sep=",", dtype=np.float32)
        raise TypeError(f"Unsupported embedding type: {type(val)}")

    arr = np.vstack([_normalize_embedding(e) for e in embs])
    return np.array(ids), arr