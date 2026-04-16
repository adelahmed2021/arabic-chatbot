import os
import cohere
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2(api_key=COHERE_API_KEY)


def get_text_embedding(text: str):
    text = text.strip()
    response = co.embed(
        model="embed-multilingual-v3.0",
        input_type="search_document",
        texts=[text],
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]


def get_query_embedding(text: str):
    text = text.strip()
    response = co.embed(
        model="embed-multilingual-v3.0",
        input_type="search_query",
        texts=[text],
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]