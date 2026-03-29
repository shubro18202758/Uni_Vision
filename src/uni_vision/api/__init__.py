"""FastAPI application factory for Uni_Vision REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from uni_vision.api.middleware import RequestLoggingMiddleware
from uni_vision.api.middleware.auth import APIKeyMiddleware
from uni_vision.api.middleware.rate_limit import RateLimitMiddleware
from uni_vision.api.middleware.security_headers import SecurityHeadersMiddleware
from uni_vision.api.routes import health, sources, detections, metrics, stats
from uni_vision.api.routes.ws_events import router as ws_router, start_redis_subscriber
from uni_vision.api.routes.agent_chat import router as agent_router
from uni_vision.api.routes.ws_agent import router as ws_agent_router
from uni_vision.api.routes.pipeline_graph import router as pipeline_graph_router
from uni_vision.api.routes.risk_analysis import router as risk_analysis_router
from uni_vision.api.routes.ws_pipeline import router as ws_pipeline_router
from uni_vision.api.routes.databricks_routes import router as databricks_router
from uni_vision.api.routes.video_upload import router as video_upload_router
from uni_vision.api.routes.pipeline_process import router as pipeline_process_router
from uni_vision.common.config import AppConfig
from uni_vision.common.logging import configure_logging


def create_app(
    config: AppConfig | None = None,
    *,
    start_pipeline: bool = False,
) -> FastAPI:
    """Build and return a configured FastAPI application.

    Parameters
    ----------
    config:
        Pre-loaded application config.  When *None* a default
        ``AppConfig()`` is used (suitable for tests / local dev).
    start_pipeline:
        When *True*, the full inference pipeline is assembled via the
        DI container and started during the lifespan.  Set to *False*
        for unit tests that don't need the live pipeline.
    """
    if config is None:
        config = AppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Configure structured logging before anything else
        try:
            configure_logging(
                level=config.logging.log_level,
                fmt=config.logging.log_format,
            )
        except (AttributeError, ImportError):
            # structlog may be stubbed in test environments
            pass

        # Startup: store config in app state for dependency injection
        app.state.config = config
        app.state.pg_client = None  # lazily connected on first request
        app.state.pipeline = None
        app.state.redis_subscriber = None
        app.state.retention_task = None
        app.state.agent_coordinator = None
        app.state.manager_agent = None
        app.state.block_registry = None
        app.state.graph_engine = None

        # Initialise block registry and graph engine (always available)
        from uni_vision.orchestrator.block_registry import BlockRegistry
        from uni_vision.orchestrator.graph_engine import GraphEngine

        app.state.block_registry = BlockRegistry()
        app.state.graph_engine = GraphEngine()

        # Initialise Ollama model router for VRAM-exclusive model swapping
        from uni_vision.agent.model_router import OllamaModelRouter

        model_router = OllamaModelRouter(
            ollama_base_url=config.ollama.base_url,
            navarasa_model=config.navarasa.model if hasattr(config, "navarasa") else "uni-vision-navarasa",
            qwen_model=config.ollama.model,
        )
        app.state.model_router = model_router

        # Activate Navarasa as the default pre-launch model
        try:
            await model_router.activate_navarasa()
        except Exception:
            import logging
            logging.getLogger(__name__).warning("model_router_initial_activation_failed", exc_info=True)

        if start_pipeline:
            from uni_vision.orchestrator.container import build_pipeline

            pipeline = build_pipeline(config)
            await pipeline.start()
            app.state.pipeline = pipeline

            # Expose Manager Agent in app state for API routes
            mgr = getattr(pipeline, "_manager_agent", None)
            if mgr is not None:
                app.state.manager_agent = mgr

            # Start Redis pub/sub subscriber for WebSocket broadcasting
            try:
                task = await start_redis_subscriber(config.redis.url)
                app.state.redis_subscriber = task
            except Exception:
                import logging
                logging.getLogger(__name__).warning("redis_subscriber_start_failed", exc_info=True)

            # Start data-retention background task when enabled
            if config.retention.enabled and app.state.pipeline is not None:
                from uni_vision.storage.retention import RetentionTask

                # Traverse pipeline → dispatcher → PostgresClient
                _disp = getattr(app.state.pipeline, "_dispatcher", None)
                pg_client = getattr(_disp, "_db", None) if _disp else None
                if pg_client is not None:
                    rt = RetentionTask(config.retention, pg_client)
                    await rt.start()
                    app.state.retention_task = rt

            # Start the agentic coordinator (Phase 21)
            if config.agent.enabled:
                from uni_vision.agent.coordinator import AgentCoordinator
                from uni_vision.api.routes.ws_events import _broadcast

                _disp = getattr(app.state.pipeline, "_dispatcher", None)
                _pg = getattr(_disp, "_db", None) if _disp else None

                coordinator = AgentCoordinator(config)
                await coordinator.start(
                    pg_client=_pg,
                    pipeline=app.state.pipeline,
                    broadcast_fn=_broadcast,
                )
                # Inject dynamic graph engine and block registry into tool context
                coordinator._context.graph_engine = app.state.graph_engine
                coordinator._context.block_registry = app.state.block_registry
                app.state.agent_coordinator = coordinator

            # Initialise Databricks integrations when enabled
            if getattr(config, "databricks", None) and config.databricks.enabled:
                try:
                    from uni_vision.databricks.delta_store import DeltaLakeStore
                    from uni_vision.databricks.mlflow_tracker import InferenceTracker
                    from uni_vision.databricks.spark_analytics import SparkAnalyticsEngine
                    from uni_vision.databricks.vector_search import VectorSearchEngine

                    dcfg = config.databricks
                    ds = DeltaLakeStore(
                        table_path=dcfg.delta.table_path,
                        audit_table_path=dcfg.delta.audit_table_path,
                        partition_columns=dcfg.delta.partition_columns,
                        checkpoint_interval=dcfg.delta.checkpoint_interval,
                        vacuum_retain_hours=dcfg.delta.vacuum_retain_hours,
                    )
                    ds.initialise()
                    app.state.databricks_delta = ds

                    mt = InferenceTracker(
                        tracking_uri=dcfg.mlflow.tracking_uri,
                        experiment_name=dcfg.mlflow.experiment_name,
                        log_every_n_frames=dcfg.mlflow.log_every_n_frames,
                        log_system_metrics=dcfg.mlflow.log_system_metrics,
                    )
                    mt.initialise()
                    app.state.databricks_mlflow = mt

                    se = SparkAnalyticsEngine(
                        master=dcfg.spark.master,
                        app_name=dcfg.spark.app_name,
                        driver_memory=dcfg.spark.driver_memory,
                        executor_memory=dcfg.spark.executor_memory,
                        delta_table_path=dcfg.spark.delta_table_path or dcfg.delta.table_path,
                    )
                    se.initialise()
                    app.state.databricks_spark = se

                    vs = VectorSearchEngine(
                        index_path=dcfg.faiss.index_path,
                        metadata_path=dcfg.faiss.metadata_path,
                        embedding_model=dcfg.faiss.embedding_model,
                        embedding_dim=dcfg.faiss.embedding_dim,
                        nprobe=dcfg.faiss.nprobe,
                        top_k=dcfg.faiss.top_k,
                        similarity_threshold=dcfg.faiss.similarity_threshold,
                    )
                    vs.initialise()
                    app.state.databricks_vector = vs

                    logging.getLogger(__name__).info("databricks_integrations_started")
                except ImportError:
                    logging.getLogger(__name__).warning(
                        "databricks_imports_unavailable — install with: pip install 'uni-vision[databricks]'"
                    )
                except Exception:
                    logging.getLogger(__name__).warning("databricks_init_failed", exc_info=True)

        yield

        # Shutdown: stop Databricks services
        db_mlflow = getattr(app.state, "databricks_mlflow", None)
        if db_mlflow is not None:
            db_mlflow.shutdown()
        db_spark = getattr(app.state, "databricks_spark", None)
        if db_spark is not None:
            db_spark.shutdown()
        db_vector = getattr(app.state, "databricks_vector", None)
        if db_vector is not None:
            db_vector.shutdown()

        # Shutdown: stop agent coordinator
        agent_coord = getattr(app.state, "agent_coordinator", None)
        if agent_coord is not None:
            await agent_coord.shutdown()

        # Shutdown: close model router
        mr = getattr(app.state, "model_router", None)
        if mr is not None:
            await mr.close()

        # Shutdown: stop retention task
        rt = getattr(app.state, "retention_task", None)
        if rt is not None:
            await rt.stop()

        # Shutdown: stop Redis subscriber
        sub_task = getattr(app.state, "redis_subscriber", None)
        if sub_task is not None:
            sub_task.cancel()

        # Shutdown: stop pipeline if running
        if app.state.pipeline is not None:
            await app.state.pipeline.shutdown()

        # Close DB pool if it was opened
        pg: object | None = getattr(app.state, "pg_client", None)
        if pg is not None and hasattr(pg, "close"):
            await pg.close()  # type: ignore[union-attr]

    app = FastAPI(
        title="Uni_Vision API",
        version="0.1.0",
        description="Real-time ANPR pipeline management and query API",
        lifespan=lifespan,
    )

    # ── Middleware (outermost first) ──────────────────────────────
    # Security headers on every response
    app.add_middleware(SecurityHeadersMiddleware)

    # CORS — restrict origins in production
    origins = [o.strip() for o in config.api.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type", "Accept"],
    )

    # Rate limiting
    app.add_middleware(RateLimitMiddleware, requests_per_minute=config.api.rate_limit_rpm)

    # API key authentication
    api_keys = {k.strip() for k in config.api.api_keys.split(",") if k.strip()}
    app.add_middleware(APIKeyMiddleware, api_keys=api_keys or None)

    # Request logging (innermost — logs after auth)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Route registration ────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(sources.router)
    app.include_router(detections.router)
    app.include_router(metrics.router)
    app.include_router(stats.router)
    app.include_router(ws_router)
    app.include_router(agent_router)
    app.include_router(ws_agent_router)
    app.include_router(pipeline_graph_router)
    app.include_router(risk_analysis_router)
    app.include_router(ws_pipeline_router)
    app.include_router(databricks_router)
    app.include_router(video_upload_router)
    app.include_router(pipeline_process_router)

    return app