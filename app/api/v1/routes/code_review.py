
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.code_review import CodeReviewRequest, CodeReviewResponse
from app.services.code_review import CodeReviewError, review_code


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/code-review",
    tags=["Code Review"],
)


@router.post(
    "",
    response_model=CodeReviewResponse,
    status_code=status.HTTP_200_OK,
)
async def create_code_review(
    payload: CodeReviewRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> CodeReviewResponse:
    """Review submitted code without modifying the original source."""

    try:
        return await review_code(payload)

    except AIProviderError as exc:
        logger.warning("Code review AI provider request failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except CodeReviewError as exc:
        logger.warning("Code review returned an invalid result.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected code review failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Code review failed unexpectedly.",
        ) from None