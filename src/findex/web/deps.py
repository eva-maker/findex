from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.models import Index
from findex.semantic import Embedder, SemanticIndex
from findex.web.settings import Settings


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_index(request: Request) -> Index:
    index: Index | None = getattr(request.app.state, "index", None)
    if index is None:
        # Індекс ще не завантажено (або lifespan не запускався): чесно кажемо «зараз недоступно».
        raise HTTPException(status_code=503, detail="індекс ще не завантажено")
    return index


IndexDep = Annotated[Index, Depends(get_index)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_semantic(request: Request) -> tuple[SemanticIndex, Embedder] | None:
    """None — векторів немає: режими semantic/hybrid тоді недоступні."""
    sem: SemanticIndex | None = getattr(request.app.state, "semantic", None)
    embedder: Embedder | None = getattr(request.app.state, "embedder", None)
    return (sem, embedder) if sem is not None and embedder is not None else None


SemanticDep = Annotated[tuple[SemanticIndex, Embedder] | None, Depends(get_semantic)]
