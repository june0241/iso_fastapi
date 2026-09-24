from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import iso_router
from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.qdrant_service import qdrant_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to ensure database collections and indexes on startup."""
    print(f"Starting {settings.PROJECT_NAME}...")
    qdrant_service.ensure_collection()
    yield
    print(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description=(
        "FastAPI service for vectorizing and indexing atomic Odoo ISO reference facts "
        "with rich metadata headings into Qdrant vector database."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """Verify service status and Qdrant connectivity."""
    q_status = qdrant_service.check_health()
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version="1.0.0",
        qdrant_status=q_status
    )


# Include API Routers
app.include_router(iso_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
