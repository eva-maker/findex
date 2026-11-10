from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    mode: str = "keyword"
    total: int = Field(description="Скільки документів підходить під запит (до пагінації).")
    page: int
    k: int = Field(description="Розмір сторінки.")
    took_ms: float
    results: list[SearchHit]


class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    length: int = Field(description="Кількість токенів.")
    text: str


class TermCount(BaseModel):
    term: str
    count: int


class StatsResponse(BaseModel):
    doc_count: int
    token_count: int
    vocab_size: int
    avg_doc_length: float
    has_positions: bool
    top_terms: list[TermCount]
    uptime_s: float


class HealthResponse(BaseModel):
    status: str
