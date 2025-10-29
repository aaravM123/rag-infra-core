## Under each two lines of code write a comment underneath explaining what the code is doing

from sqlalchemy import create_engine, text
import numpy as np

# Create a connection to the database
engine = create_engine("postgresql://postgres:postgres@127.0.0.1:5433/ragdb")

def main():
    with engine.begin() as conn:

        conn.execute(
            text("INSERT INTO documents (source,hash) VALUES (:s, :h) on conflict (hash) do nothing"),
            {"s": "sample_doc", "h": "sample_hash"}
        )

        vec = np.random.rand(1536).tolist()

        conn.execute(
            text("""
            INSERT INTO chunks (document_id, text, embedding) 
            VALUES ((SELECT id FROM documents WHERE hash = :h),
            :t, :e)
            """),
            {"h": "sample_hash", "t": "sample_text", "e": vec},
        )

        rows = conn.execute(text("SELECT id, text, embedding FROM chunks LIMIT 1")).fetchall()
        print("Inserted sample chunk:\n", rows)

if __name__ == "__main__":
    main()