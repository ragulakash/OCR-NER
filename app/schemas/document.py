from typing import Any, Optional

from pydantic import BaseModel, Field


class JSONDocument(BaseModel):
    document_id: str = Field(min_length=1)
    content: dict[str, Any]
    source: Optional[str] = None


class ExtractedDocument(BaseModel):
    document_id: str
    text: str = Field(min_length=1)
    source_type: str