from typing import Annotated

from fastapi import Depends, HTTPException, Request

from findex.models import Index
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
