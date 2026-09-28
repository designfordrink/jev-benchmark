from __future__ import annotations


def model_size_bytes(policy) -> int | None:
    value = getattr(policy, "model_size_bytes_fp32", None)
    return int(value) if value is not None else None
