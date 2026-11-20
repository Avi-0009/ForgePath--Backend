from fastapi import APIRouter

from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.code_review import router as code_review_router
from app.api.v1.routes.local_context import router as local_context_router
from app.api.v1.routes.code_patch import router as code_patch_router
from app.api.v1.routes.workspace import router as workspace_router
from app.api.v1.routes.execution import router as test_execution_router
from app.api.v1.routes.task_planning import router as task_planning_router
from app.api.v1.routes.agent_orchestration import router as agent_orchestration_router
from app.api.v1.routes.ai_usage import router as ai_usage_router
from app.api.v1.routes.projects import router as projects_router
from app.api.v1.routes.ai_configuration import router as ai_configuration_router
from app.api.v1.routes.project_snapshots import router as project_snapshots_router
from app.api.v1.routes.project_reviews import router as project_reviews_router
from app.api.v1.routes.project_review_tasks import router as project_review_tasks_router
from app.api.v1.routes.project_code_search import router as project_code_search_router
from app.api.v1.routes.project_investigations import router as project_investigations_router


router = APIRouter(prefix="/api/v1")

router.include_router(auth_router)
router.include_router(code_review_router)
router.include_router(local_context_router)
router.include_router(code_patch_router)
router.include_router(workspace_router)
router.include_router(test_execution_router)
router.include_router(task_planning_router)
router.include_router(agent_orchestration_router)
router.include_router(ai_usage_router)
router.include_router(projects_router)
router.include_router(ai_configuration_router)
router.include_router(project_snapshots_router)
router.include_router(project_reviews_router)
router.include_router(project_review_tasks_router)
router.include_router(project_code_search_router)
router.include_router(project_investigations_router)