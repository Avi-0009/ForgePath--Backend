import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.code_patch import (
    CodePatchRequest,
    CodePatchResponse,
)
from app.services.code_patch import (
    CodePatchError,
    StaleSourceFileError,
    generate_code_patch,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/code-patch",
    tags=["Code Patch"],
)


@router.post(
    "",
    response_model=CodePatchResponse,
    status_code=status.HTTP_200_OK,
)
async def create_code_patch(
    payload: CodePatchRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> CodePatchResponse:
    """
    Generate a proposed code fix without modifying the source file.
    """

    try:
        return await generate_code_patch(payload)

    except StaleSourceFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except AIProviderError as exc:
        logger.warning("AI provider failed during patch generation.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except CodePatchError as exc:
        logger.warning("Could not generate a valid code patch.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected patch-generation failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Patch generation failed unexpectedly.",
        ) from None