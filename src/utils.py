import os
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def add_documents_to_postgres(urls, chunk_numbers, contents, metadatas, url_to_full_document=None, batch_size=20):
    """
    Add documentation chunks to Postgres with pgvector.
    """
    with Session() as session:
        for i in range(0, len(urls), batch_size):
            for url, chunk_number, content, metadata in zip(
                urls[i:i+batch_size],
                chunk_numbers[i:i+batch_size],
                contents[i:i+batch_size],
                metadatas[i:i+batch_size]
            ):
                embedding = model.encode(content).tolist()
                session.execute(
                    text("""
                        INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, embedding)
                        VALUES (:url, :chunk_number, :content, :metadata, :source_id, :embedding)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "url": url,
                        "chunk_number": chunk_number,
                        "content": content,
                        "metadata": json.dumps(metadata),
                        "source_id": metadata.get("source"),
                        "embedding": embedding
                    }
                )
        session.commit()

def add_code_examples_to_postgres(urls, chunk_numbers, examples, summaries, metadatas, batch_size=20):
    """
    Add code example chunks to Postgres with pgvector.
    """
    with Session() as session:
        for i in range(0, len(urls), batch_size):
            for url, chunk_number, code, summary, metadata in zip(
                urls[i:i+batch_size],
                chunk_numbers[i:i+batch_size],
                examples[i:i+batch_size],
                summaries[i:i+batch_size],
                metadatas[i:i+batch_size]
            ):
                embedding = model.encode(code).tolist()
                session.execute(
                    text("""
                        INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, embedding)
                        VALUES (:url, :chunk_number, :content, :summary, :metadata, :source_id, :embedding)
                        ON CONFLICT DO NOTHING
                    """),
                    {
                        "url": url,
                        "chunk_number": chunk_number,
                        "content": code,
                        "summary": summary,
                        "metadata": json.dumps(metadata),
                        "source_id": metadata.get("source"),
                        "embedding": embedding
                    }
                )
        session.commit()

def search_documents_postgres(query, match_count=5, filter_metadata=None):
    """
    Vector search for documentation chunks in Postgres.
    """
    embedding = model.encode(query).tolist()
    sql = """
        SELECT *, 1 - (embedding <=> :embedding) AS similarity
        FROM crawled_pages
        WHERE (:source_id IS NULL OR source_id = :source_id)
        ORDER BY embedding <=> :embedding
        LIMIT :match_count
    """
    source_id = filter_metadata.get("source") if filter_metadata else None
    with Session() as session:
        res = session.execute(
            text(sql),
            {"embedding": embedding, "source_id": source_id, "match_count": match_count}
        )
        results = []
        for row in res:
            row_dict = dict(row)
            results.append(row_dict)
        return results

def search_code_examples_postgres(query, match_count=5, filter_metadata=None):
    """
    Vector search for code examples in Postgres.
    """
    embedding = model.encode(query).tolist()
    sql = """
        SELECT *, 1 - (embedding <=> :embedding) AS similarity
        FROM code_examples
        WHERE (:source_id IS NULL OR source_id = :source_id)
        ORDER BY embedding <=> :embedding
        LIMIT :match_count
    """
    source_id = filter_metadata.get("source") if filter_metadata else None
    with Session() as session:
        res = session.execute(
            text(sql),
            {"embedding": embedding, "source_id": source_id, "match_count": match_count}
        )
        results = []
        for row in res:
            row_dict = dict(row)
            results.append(row_dict)
        return results

def update_source_info_postgres(source_id, summary, total_words):
    """
    Upsert source info in the sources table.
    """
    with Session() as session:
        session.execute(
            text("""
                INSERT INTO sources (source_id, summary, total_words)
                VALUES (:source_id, :summary, :total_words)
                ON CONFLICT (source_id)
                DO UPDATE SET summary = EXCLUDED.summary, total_words = EXCLUDED.total_words
            """),
            {
                "source_id": source_id,
                "summary": summary,
                "total_words": total_words
            }
        )
        session.commit()

# The following are placeholders for your actual implementations.
def extract_code_blocks(markdown):
    # Implement your logic for extracting code blocks with context
    return []

def generate_code_example_summary(code, context_before, context_after):
    # Implement your summary generation logic, e.g., using LLM or heuristic
    return "Code summary."

def extract_source_summary(source_id, content):
    # Implement your summary logic for the source/domain
    return f"Summary for {source_id}"

# If you have any utility functions from the old utils.py you want to keep, add them here!
