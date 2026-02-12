from pydantic import BaseModel
from typing import Optional, List

# --- existing models ---
class SearchRequest(BaseModel):
    query: str
    n: int = 30

class IngestItem(BaseModel):
    title: Optional[str] = None
    landing_page_url: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None

class IngestRequest(BaseModel):
    items: List[IngestItem]

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

# --- NEW: OpenAlex flow models ---
class OpenAlexSearchRequest(BaseModel):
    query: str
    n: int = 20
    lang: str = "ru"      # ru/en
    only_oa: bool = True  # только open access

class OpenAlexItem(BaseModel):
    openalex_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    landing_page_url: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_ok: bool = False
    pdf_reason: Optional[str] = None
    cited_by_count: Optional[int] = None

class OpenAlexImportRequest(BaseModel):
    items: List[OpenAlexItem]

