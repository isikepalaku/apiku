from fastapi import FastAPI, Depends, HTTPException
from starlette.middleware.cors import CORSMiddleware
from upstash_ratelimit import Ratelimit, FixedWindow
from upstash_redis import Redis
from fastapi import Request
from functools import partial
import time

from api.settings import api_settings
from api.routes.v1_router import v1_router
from api.dependencies.auth import verify_api_key

# Initialize Redis client
redis = Redis.from_env()

# Create a rate limiter that allows 10 requests per minute
ratelimit = Ratelimit(
    redis=redis,
    limiter=FixedWindow(max_requests=10, window=60),  # 10 requests per 60 seconds (1 minute)
    prefix="@upstash/ratelimit"
)

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware - limits to 10 requests per minute per IP"""
    # Use client IP as identifier
    identifier = request.client.host if request.client else "default"
    response = ratelimit.limit(identifier)

    if not response.allowed:
        # Calculate time until reset
        reset_after = response.reset - time.time()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded - maximum 10 requests per minute",
                "reset_after": reset_after,
                "remaining": response.remaining
            }
        )

    return await call_next(request)

def create_app() -> FastAPI:
    """Create a FastAPI App

    Returns:
        FastAPI: FastAPI App
    """

    # Create FastAPI App
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
    )

    # Add v1 router with API key dependency
    app.include_router(v1_router, dependencies=[Depends(verify_api_key)])

    # Add Middlewares
    app.middleware("http")(rate_limit_middleware)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

# Create FastAPI app
app = create_app()

@app.get("/health")
async def health_check():
    """Health check endpoint that is rate limited"""
    return {"status": "healthy"}