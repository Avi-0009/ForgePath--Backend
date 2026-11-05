import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Request

from app.api.dependencies import get_current_user
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.code_patch import (
    CodePatchRequest,
    CodePatchResponse,
    CodePatchApplyRequest,
    CodePatchApplyResponse,
    CodePatchRollbackRequest,
    CodePatchRollbackResponse,
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
    PatchBackupNotFound,
    PatchBackupIntegrityError,
    rollback_code_patch,
)
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceBinaryFile,
    WorkspaceConfigurationError,
    WorkspaceFileNotFound,
    WorkspaceFileTooLarge,
    WorkspaceInvalidPath,
    WorkspaceReadError,
    read_configured_workspace_file,
)
from app.services.workspace_review_context import (
    WorkspaceReviewContextBuilder,
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
    request: Request,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> CodePatchResponse:
    """Generate a proposed code fix without modifying the source file."""

    try:
        related_workspace_context = None

        indexing_service = getattr(
            request.app.state,
            "workspace_indexing_service",
            None,
        )

        if indexing_service is not None:

            # Read the selected source safely to construct the retrieval query.
            source = read_configured_workspace_file(
                workspace_root=payload.workspace_root,
                relative_path=payload.relative_path,
            )

            # Retrieve verified related workspace code.
            try:
                context_builder = WorkspaceReviewContextBuilder(
                    indexing_service=indexing_service,
                )

                workspace_context = await context_builder.build(
                    workspace_root=payload.workspace_root,
                    relative_path=source.relative_path,
                    source_content=source.content,
                    language=source.language,
                )

                if not workspace_context.is_empty:
                    related_workspace_context = (
                        workspace_context.to_prompt_text()
                    )

                    logger.info(
                        "Retrieved %d verified workspace context chunks "
                        "(%d characters) for patch generation.",
                        workspace_context.chunk_count,
                        workspace_context.total_chars,
                    )

            except Exception:
                # Retrieval is optional. Patch generation can continue
                # using the selected source and saved project instructions.
                logger.exception(
                    "Workspace context retrieval failed. "
                    "Continuing without RAG context."
                )

        return await generate_code_patch(
            payload,
            related_workspace_context=related_workspace_context,
        )

    except StaleSourceFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except WorkspaceConfigurationError as exc:
        logger.error("Patch generation workspace configuration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Workspace configuration is invalid or unavailable.",
        ) from exc

    except WorkspaceAccessDenied as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    except WorkspaceInvalidPath as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except WorkspaceFileNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except WorkspaceFileTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc

    except WorkspaceBinaryFile as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    except WorkspaceReadError:
        logger.exception(
            "Failed to read a workspace file during patch generation."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The requested file could not be read.",
        ) from None

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


@router.post(
    "/rollback",
    response_model=CodePatchRollbackResponse,
    status_code=status.HTTP_200_OK,
)
async def rollback_applied_code_patch(
    payload: CodePatchRollbackRequest,
    _current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
) -> CodePatchRollbackResponse:
    """Restore the original file from a verified patch backup."""

    try:
        return await rollback_code_patch(payload)

    except PatchBackupNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except PatchBackupIntegrityError as exc:
        logger.error("Patch backup integrity verification failed.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except StalePatchSourceError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except WorkspaceConfigurationError as exc:
        logger.error("Rollback workspace configuration error: %s", exc)
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
        logger.exception("Patch rollback failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected patch rollback failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected rollback error occurred.",
        ) from None