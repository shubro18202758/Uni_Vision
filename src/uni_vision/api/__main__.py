"""``python -m uni_vision.api`` — start the REST API via uvicorn."""

from __future__ import annotations

import uvicorn

from uni_vision.api import create_app
from uni_vision.common.config import load_config


def main() -> None:
    config = load_config()
    app = create_app(config, start_pipeline=True)
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
