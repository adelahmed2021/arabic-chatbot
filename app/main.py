from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.db import engine, Base, SessionLocal
from app import models
from app.schemas import AskRequest, AskResponse
from app.crud import create_document, create_chunk
from app.embeddings import get_text_embedding
from app.rag import chunk_text_by_subject, answer_with_rag
from app.config import VALID_SUBJECTS

app = FastAPI(title="Arabic Chatbot API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"message": "Arabic Chatbot API is running"}


@app.post("/ingest-file")
async def ingest_file(
    subject: str = Form(...),
    source_type: str = Form("base"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if subject not in VALID_SUBJECTS:
        raise HTTPException(status_code=400, detail="Invalid subject")

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    content_bytes = await file.read()

    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            content = content_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    content = content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    

    doc = create_document(
        db=db,
        title=file.filename,
        subject=subject,
        source_type=source_type
    )

    chunks = chunk_text_by_subject(content, subject)

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks generated")

    for i, chunk in enumerate(chunks):
        emb = get_text_embedding(chunk)
        create_chunk(
            db=db,
            document_id=doc.id,
            subject=subject,
            text_value=chunk,
            order=i,
            embedding=emb
        )

    db.commit()

    return {
        "message": "File ingested successfully",
        "filename": file.filename,
        "subject": subject,
        "chunks": len(chunks)
    }


@app.post("/ask", response_model=AskResponse)
def ask(data: AskRequest, db: Session = Depends(get_db)):
    result = answer_with_rag(db, data.question)
    return result