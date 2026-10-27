import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.code_patch import (
    CodePatchRequest,
    CodePatchResponse,
    CodePatchApplyRequest,
    CodePatchApplyResponse,
)
from app.services.code_patch import (
    CodePatchError,
    StaleSourceFileError,
    generate_code_patch,
)
from app.services.code_patch_apply import (
    PatchApplyError,
    ProposedPatchHashMismatch,
    StalePatchSourceError,
    apply_code_patch,
)
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceConfigurationError,
    WorkspaceFileNotFound,
    WorkspaceInvalidPath,
    WorkspaceReadError,
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


@router.post(
    "/apply",
    response_model=CodePatchApplyResponse,
    status_code=status.HTTP_200_OK,
)
async def apply_approved_code_patch(
    payload: CodePatchApplyRequest,
    _current_user: Annotated[
        User, Depends(get_current_user)
    ],
) -> CodePatchApplyResponse:
    """Apply a patch only after explicit approval and source verification."""
    try:
        return await apply_code_patch(payload)

    except StalePatchSourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ProposedPatchHashMismatch as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except WorkspaceConfigurationError as exc:
        logger.error("Patch application configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid.",
        ) from exc

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except WorkspaceFileNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except WorkspaceInvalidPath as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except WorkspaceReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except PatchApplyError as exc:
        logger.exception("Patch application failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error while applying patch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while applying the patch.",
        ) from exc