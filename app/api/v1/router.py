from fastapi import APIRouter

from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.code_review import router as code_review_router
from app.api.v1.routes.local_context import router as local_context_router


router = APIRouter(prefix="/api/v1")

router.include_router(auth_router)
router.include_router(code_review_router)
router.include_router(local_context_router)