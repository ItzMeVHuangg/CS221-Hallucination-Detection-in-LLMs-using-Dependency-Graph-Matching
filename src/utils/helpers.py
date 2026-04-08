"""
Utility helpers: config loading, logging, reproducibility.
"""

import os
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML config and return as nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_nested(cfg: Dict, *keys, default=None):
    """Safely navigate nested dict: get_nested(cfg, 'model', 'device')."""
    val = cfg
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: str = "INFO") -> logging.Logger:
    """Create a named logger with console (and optional file) handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── I/O ─────────────────────────────────────────────────────────────────────

def save_json(obj: Any, path: str, indent: int = 2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs(*dirs: str):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
