FROM python:3.13-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install \
    --no-cache-dir \
    --index-url https://pypi.org/simple \
    --timeout 120 \
    --retries 10 \
    -r requirements.txt

RUN groupadd --system --gid "${APP_GID}" forgepath \
    && useradd --system --uid "${APP_UID}" --gid forgepath \
        --home-dir /var/lib/forgepath --create-home forgepath \
    && mkdir -p /var/lib/forgepath/snapshots /var/lib/forgepath/conversations \
    && chown -R forgepath:forgepath /var/lib/forgepath \
    && chmod 0700 /var/lib/forgepath \
        /var/lib/forgepath/snapshots \
        /var/lib/forgepath/conversations

COPY --chown=forgepath:forgepath app/ ./app/
COPY --chown=forgepath:forgepath alembic/ ./alembic/
COPY --chown=forgepath:forgepath alembic.ini .

USER forgepath

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
