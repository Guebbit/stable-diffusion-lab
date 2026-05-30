"""
Shared logging helper for the Stable Diffusion Lab backend.

Problem solved:
  logging.basicConfig() is a no-op if the root logger already has handlers.
  Uvicorn configures the root logger before importing the app, so basicConfig
  calls inside main.py / model_service.py have no effect.

Solution:
  Attach a StreamHandler directly to each module's own logger.
  This ensures log output reaches stdout regardless of root-logger state.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, attaching a StreamHandler if it has none yet.

    Call this in every module instead of logging.getLogger(name) directly:
      logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)s:     %(name)s - %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
