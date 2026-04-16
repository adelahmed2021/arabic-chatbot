from pydantic import BaseModel


class IngestRequest(BaseModel):
    title: str
    subject: str
    source_type: str = "base"   # base / examples
    content: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    subject: str
    answer: str
    chunks_used: int