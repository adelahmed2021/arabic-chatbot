from sqlalchemy import text
from app.models import Document, Chunk


def create_document(db, title: str, subject: str, source_type: str):
    doc = Document(
        title=title,
        subject=subject,
        source_type=source_type
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def create_chunk(db, document_id: int, subject: str, text_value: str, order: int, embedding):
    chunk = Chunk(
        document_id=document_id,
        subject=subject,
        chunk_text=text_value,
        chunk_order=order,
        embedding=embedding,
    )
    db.add(chunk)


def delete_subject_data(db, subject: str):
    db.execute(text("DELETE FROM chunks WHERE subject = :subject"), {"subject": subject})
    db.execute(text("DELETE FROM documents WHERE subject = :subject"), {"subject": subject})
    db.commit()


def search_similar_chunks(db, subject: str, embedding, limit: int = 3):
    query = text("""
        SELECT chunk_text
        FROM chunks
        WHERE subject = :subject
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = db.execute(
        query,
        {
            "subject": subject,
            "embedding": str(embedding),
            "limit": limit,
        }
    )

    return [row[0] for row in result]