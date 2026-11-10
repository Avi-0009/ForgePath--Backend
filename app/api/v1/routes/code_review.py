import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import get_current_user_with_usage_context
from app.core.ai_provider import AIProviderError
from app.models.user import User
from app.schemas.code_review import (
    CodeReviewRequest,
    CodeReviewResponse,
    CodeReviewFileRequest,
    CodeReviewFileResponse,
)
from app.services.code_review import CodeReviewError, review_code
from app.services.workspace import (
    WorkspaceAccessDenied,
    WorkspaceBinaryFile,
    WorkspaceFileNotFound,
    WorkspaceFileTooLarge,
    WorkspaceInvalidPath,
    WorkspaceReadError,
    read_configured_workspace_file,
)
from app.services.local_context import LocalContextError, get_project_instructions
from app.services.workspace_review_context import (
    WorkspaceReviewContextBuilder,
)


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
        Depends(get_current_user_with_usage_context),
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


@router.post(
    "/file",
    response_model=CodeReviewFileResponse,
    status_code=status.HTTP_200_OK,
)
async def review_local_file(
    payload: CodeReviewFileRequest,
    request: Request,
    _current_user: Annotated[
        User,
        Depends(get_current_user_with_usage_context),
    ],
) -> CodeReviewFileResponse:
    """Review a permitted local file without modifying it."""

    try:
        source = read_configured_workspace_file(
            workspace_root=payload.workspace_root,
            relative_path=payload.relative_path,
        )

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
        logger.exception("Failed to read a workspace file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The requested file could not be read.",
        ) from None

    if not source.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The selected file is empty.",
        )

    # CodeReviewRequest currently limits source code to 100,000 characters.
    if len(source.content) > 100_000:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The file exceeds the code-review limit of 100,000 characters.",
        )

    review_request = CodeReviewRequest(
        code=source.content,
        language=source.language,
        review_mode=payload.review_mode,
        task_description=payload.task_description,
        additional_context=payload.additional_context,
    )

    try:
        project_context = await get_project_instructions(
            project_root=payload.workspace_root,
        )
    except LocalContextError:
        logger.exception(
            "Failed to retrieve saved project instructions."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve saved project context.",
        ) from None

    # Retrieve optional semantic workspace context.
    related_workspace_context = None

    indexing_service = getattr(
        request.app.state,
        "workspace_indexing_service",
        None,
    )

    if indexing_service is not None:
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
                    "(%d characters) for code review.",
                    workspace_context.chunk_count,
                    workspace_context.total_chars,
                )

            else:
                logger.debug(
                    "No verified related workspace context was found."
                )

        except Exception:
            # RAG is an optional enhancement. A retrieval failure
            # must not prevent ordinary code review.
            logger.exception(
                "Workspace context retrieval failed. "
                "Continuing without RAG context."
            )

    try:
        review = await review_code(
            review_request,
            project_instructions=(
                project_context.instructions
                if project_context is not None
                else None
            ),
            related_workspace_context=related_workspace_context,
        )

    except AIProviderError as exc:
        logger.warning("AI provider failed during local-file review.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except CodeReviewError as exc:
        logger.warning("Local-file review returned an invalid result.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    except Exception:
        logger.exception("Unexpected local-file review failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Code review failed unexpectedly.",
        ) from None

    return CodeReviewFileResponse(
        relative_path=source.relative_path,
        language=source.language,
        sha256=source.sha256,
        size_bytes=source.size_bytes,
        line_count=source.line_count,
        review=review,
    )
