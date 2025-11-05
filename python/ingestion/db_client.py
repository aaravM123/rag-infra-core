import psycopg2
import numpy as np
import os
DB_CONFIG = dict(
    dbname = "ragdb",
    user = "postgres",
    password = "postgres",
    host="localhost",
    port="5432",
)

def fetch_embeddings(limit=1000):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT id, embedding FROM chunks LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
 

    ids, embs = zip(*rows)
    arr = np.vstack(embs).astype(np.float32)
    return np.array(ids), arr