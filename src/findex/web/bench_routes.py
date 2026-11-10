import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from findex.models import Index
from findex.query import QueryError
from findex.ranking import ScorerName
from findex.web.deps import get_index, get_settings
from findex.web.service import run_search
from findex.web.settings import Settings

router = APIRouter(prefix="/_bench", include_in_schema=False)


@router.get("/search-async")
async def search_async(
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    q: Annotated[str, Query(min_length=1)],
    k: int = 10,
    scorer: ScorerName = "bm25",
) -> dict[str, int]:
    try:
        page = run_search(index, settings.corpus_dir, q, k, 1, scorer)  # блокує цикл подій!
    except QueryError:
        return {"total": 0}
    return {"total": page.total}


@router.get("/search-to-thread")
async def search_to_thread(
    index: Annotated[Index, Depends(get_index)],
    settings: Annotated[Settings, Depends(get_settings)],
    q: Annotated[str, Query(min_length=1)],
    k: int = 10,
    scorer: ScorerName = "bm25",
) -> dict[str, int]:
    try:
        page = await asyncio.to_thread(run_search, index, settings.corpus_dir, q, k, 1, scorer)
    except QueryError:
        return {"total": 0}
    return {"total": page.total}
