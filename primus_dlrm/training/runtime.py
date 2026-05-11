"""Runtime configuration for precision settings and CLI overrides."""
from __future__ import annotations

import logging
import os
from dataclasses import asdict

import torch
import yaml

from primus_dlrm.config import Config, TrainConfig

logger = logging.getLogger(__name__)

# Known env-var overrides for fields that are read from the environment at
# wrap time (instead of mutating the config object). Used by
# ``log_resolved_config`` to annotate the dumped YAML so the log shows the
# *effective* value alongside the YAML default and the env source.
#
# Each entry maps a config dotted path to (env_var_name, value_coercer).
_ENV_OVERRIDE_FIELDS: dict[tuple[str, ...], tuple[str, type]] = {
    ("distributed", "embedding_sharding", "topology", "hbm_cap_gb"):
        ("PRIMUS_TORCHREC_HBM_CAP_GB", float),
    ("distributed", "embedding_sharding", "topology", "ddr_cap_gb"):
        ("PRIMUS_TORCHREC_DDR_CAP_GB", float),
    ("distributed", "embedding_sharding", "topology", "local_world_size"):
        ("PRIMUS_TORCHREC_LOCAL_WORLD_SIZE", int),
}


def configure_runtime(tc: TrainConfig) -> None:
    """Set global precision flags based on training config.

    Call once at startup, before any computation. Configures:
    - BF16/TF32 precision modes
    """
    torch.backends.cuda.matmul.allow_tf32 = tc.allow_tf32
    torch.backends.cudnn.allow_tf32 = tc.allow_tf32

    logger.info(
      f"{torch.backends.cuda.matmul.allow_tf32=}, \
        {torch.backends.cudnn.allow_tf32=}"
    )


def apply_cli_overrides(config: Config, args) -> Config:
    """Apply CLI argument overrides to the Config object.

    Records per-field overrides on ``config._cli_overrides`` (a dict keyed by
    dotted path tuple) so ``log_resolved_config`` can annotate the YAML dump
    with their source and the original YAML value.
    """
    cli_overrides: dict[tuple[str, ...], dict] = {}

    def _maybe(path: tuple[str, ...], new_val, src: str, get, set_):
        old_val = get()
        if old_val == new_val:
            return  # CLI value matches YAML; not a real override
        cli_overrides[path] = {"yaml": old_val, "new": new_val, "src": src}
        set_(new_val)

    if hasattr(args, "dense_strategy"):
        _maybe(
            ("distributed", "dense_strategy"),
            args.dense_strategy,
            f"--dense-strategy={args.dense_strategy}",
            lambda: config.distributed.dense_strategy,
            lambda v: setattr(config.distributed, "dense_strategy", v),
        )
    if hasattr(args, "embedding_sharding"):
        _maybe(
            ("distributed", "embedding_sharding", "strategy"),
            args.embedding_sharding,
            f"--embedding-sharding={args.embedding_sharding}",
            lambda: config.distributed.embedding_sharding.strategy,
            lambda v: setattr(config.distributed.embedding_sharding, "strategy", v),
        )
    if getattr(args, "attention_impl", None):
        _maybe(
            ("model", "transformer", "attention_impl"),
            args.attention_impl,
            f"--attention-impl={args.attention_impl}",
            lambda: config.model.transformer.attention_impl,
            lambda v: setattr(config.model.transformer, "attention_impl", v),
        )

    if cli_overrides:
        summary = ", ".join(f"{'.'.join(p)}={d['new']}" for p, d in cli_overrides.items())
        logger.info(f"CLI overrides applied: {summary}")

    # Stash for log_resolved_config to consume; private attribute (underscore
    # prefix) so it doesn't show up in ``asdict()``.
    config._cli_overrides = cli_overrides  # type: ignore[attr-defined]
    return config


def _detect_env_overrides(effective: dict) -> dict[tuple[str, ...], dict]:
    """Detect known env-var overrides; mutate ``effective`` to show the env value."""
    overrides: dict[tuple[str, ...], dict] = {}
    for path, (env_name, coerce) in _ENV_OVERRIDE_FIELDS.items():
        raw = os.environ.get(env_name, "")
        raw = raw.strip() if raw else ""
        if not raw:
            continue
        try:
            new_val = coerce(raw)
        except (TypeError, ValueError):
            continue
        # Walk ``effective`` to read the YAML value at this path; tolerate
        # missing keys (e.g. legacy configs without a topology block).
        node = effective
        ok = True
        for k in path[:-1]:
            if not isinstance(node, dict) or k not in node:
                ok = False
                break
            node = node[k]
        if not ok:
            continue
        old_val = node.get(path[-1])
        if old_val == new_val:
            continue  # env matches YAML; not a real override
        node[path[-1]] = new_val  # mutate so the dump shows the env value
        overrides[path] = {"yaml": old_val, "new": new_val,
                           "src": f"env {env_name}={raw}"}
    return overrides


def _annotate_yaml_lines(rendered: str,
                          overrides: dict[tuple[str, ...], dict]) -> str:
    """Append ``# OVERRIDDEN ...`` comments to leaf lines whose path is in ``overrides``.

    Tracks the current dotted path by indentation depth. Skips list-of-dicts
    entries (path tracking is for fixed-key mappings only); the known overrides
    today live entirely under fixed paths so this is fine.
    """
    if not overrides:
        return rendered
    out: list[str] = []
    path_stack: list[str] = []
    indent_stack: list[int] = []
    for line in rendered.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        # Pop path elements at >= current indent (we're back out at this level).
        while indent_stack and indent_stack[-1] >= indent:
            path_stack.pop()
            indent_stack.pop()
        # List items break the dotted-path tracking; emit verbatim.
        if stripped.startswith("- ") or not stripped or ":" not in stripped:
            out.append(line)
            continue
        key, _, val_part = stripped.partition(":")
        key = key.strip()
        val_part = val_part.strip()
        if val_part:
            full = tuple(path_stack + [key])
            ann = overrides.get(full)
            if ann is not None:
                line = f"{line}    # OVERRIDDEN: {ann['src']} (yaml: {ann['yaml']!r})"
            out.append(line)
        else:
            # Mapping start — push onto path.
            out.append(line)
            path_stack.append(key)
            indent_stack.append(indent)
    return "\n".join(out)


def log_resolved_config(config: Config, source_path: str | None = None) -> None:
    """Log the fully-resolved config (post-CLI + env overrides) as YAML.

    Inline-annotates fields that were overridden so the log shows BOTH the
    effective value and where the override came from. Recognized override
    sources:

    - **CLI**: any ``--flag`` handled by ``apply_cli_overrides`` whose value
      differed from the YAML.
    - **env**: ``PRIMUS_TORCHREC_HBM_CAP_GB`` / ``_DDR_CAP_GB`` /
      ``_LOCAL_WORLD_SIZE`` for the ``topology`` block.

    Only emits on the main process so multi-rank logs aren't spammed 8x.
    """
    from primus_dlrm.distributed.setup import is_main_process
    if not is_main_process():
        return

    effective = asdict(config)
    overrides: dict[tuple[str, ...], dict] = {}
    overrides.update(getattr(config, "_cli_overrides", {}) or {})
    overrides.update(_detect_env_overrides(effective))

    rendered = yaml.dump(
        effective, default_flow_style=False, sort_keys=False, width=120,
    )
    annotated = _annotate_yaml_lines(rendered, overrides)

    header = f"=== Resolved config (source={source_path}) ===" if source_path \
        else "=== Resolved config ==="
    logger.info(header)
    for line in annotated.splitlines():
        logger.info(line)
    if overrides:
        logger.info(f"--- {len(overrides)} override(s) applied ---")
        for path, info in overrides.items():
            logger.info(
                f"    {'.'.join(path)}  yaml={info['yaml']!r} -> {info['new']!r}  "
                f"({info['src']})"
            )
    logger.info("=" * len(header))
