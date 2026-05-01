"""Shared utilities for primus_dlrm.

Sub-modules:
  - model_params: parameter counting + size estimation + startup logging
"""
from primus_dlrm.utils.model_params import *  # noqa: F401,F403
from primus_dlrm.utils import model_params

__all__ = list(model_params.__all__)
