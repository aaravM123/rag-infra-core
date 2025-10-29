CREATE TABLE IF NOT EXISTS documents(
    id SERIAL PRIMARY KEY,
    source TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    hash TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS chunks(
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id),
    text TEXT,
    embedding vector(1536)
);