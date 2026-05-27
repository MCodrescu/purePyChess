"""HuggingFace Hub weight management for purePyChess.

On first Engine instantiation the weights are downloaded automatically from the
configured HF repository and cached at ``~/.cache/purepychess/weights.pt``.
Subsequent instantiations load from the local cache with no network call.

Usage
-----
    from purepychess.weights import load_weights, download_weights

    model = ChessNet()
    load_weights(model)          # auto-downloads if necessary

    download_weights()           # explicit pre-cache
"""

from __future__ import annotations

from pathlib import Path

import torch

from purepychess.config import HF_REPO_ID

_CACHE_DIR   = Path.home() / ".cache" / "purepychess"
_WEIGHTS_FILE = _CACHE_DIR / "weights.pt"


def get_weights_path() -> Path:
    """Return the path to cached weights, downloading from HF Hub if absent."""
    if _WEIGHTS_FILE.exists():
        return _WEIGHTS_FILE

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
        import shutil

        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="weights.pt",
            cache_dir=str(_CACHE_DIR),
        )
        shutil.copy2(downloaded, _WEIGHTS_FILE)
        return _WEIGHTS_FILE

    except Exception as exc:
        raise RuntimeError(
            f"Could not download weights. Check internet connection or manually "
            f"place weights.pt in {_CACHE_DIR}. "
            f"Download from: https://huggingface.co/{HF_REPO_ID}\n"
            f"Original error: {exc}"
        ) from exc


def download_weights() -> Path:
    """Pre-cache model weights from HuggingFace Hub. Returns the local path."""
    return get_weights_path()


def load_weights(
    model: torch.nn.Module,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load weights into *model* in-place. Returns *model* for convenience."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = get_weights_path()
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
