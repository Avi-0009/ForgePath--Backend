"""PostgreSQL leases for bounded review attempts; never automatically retry AI."""
from datetime import timedelta
from uuid import UUID

from sqlalchemy import func, or_, update
from app.models.project_review import ProjectReview

LEASE_SECONDS = 600
CALL_TIMEOUT_SECONDS = 240
INTERRUPTED_MESSAGE = "Review interrupted. Confirm a new review to retry; additional AI charges may apply."


async def expire_reviews(db, user_id: UUID) -> None:
    await db.execute(update(ProjectReview).where(
        ProjectReview.user_id == user_id,
        ProjectReview.status.in_(("queued", "running")),
        or_(ProjectReview.lease_expires_at.is_(None), ProjectReview.lease_expires_at <= func.now()),
    ).values(status="interrupted", lease_expires_at=None, error_message=INTERRUPTED_MESSAGE))


async def write_owned_review(db, review_id, lease_token, *, claim=False, **values) -> bool:
    """Fence delayed/duplicate workers using status, token and database time."""
    terminal = values.get("status") in {"completed", "partial", "failed", "interrupted"}
    result = await db.execute(update(ProjectReview).where(
        ProjectReview.id == review_id,
        ProjectReview.lease_token == lease_token,
        ProjectReview.status == ("queued" if claim else "running"),
        ProjectReview.lease_expires_at > func.now(),
    ).values(**values, lease_expires_at=(None if terminal else func.now() + timedelta(seconds=LEASE_SECONDS))))
    await db.commit()
    return result.rowcount == 1
