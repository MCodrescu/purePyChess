"""Tests for purepychess.weights."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from purepychess.model import ChessNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_random_weights(path: Path) -> None:
    """Save a valid ChessNet state_dict to *path*."""
    model = ChessNet()
    torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# get_weights_path
# ---------------------------------------------------------------------------

class TestGetWeightsPath:
    def test_returns_existing_file_without_download(self, tmp_path):
        """If weights.pt already exists in cache, no download is attempted."""
        fake_cache = tmp_path / "purepychess"
        fake_cache.mkdir()
        weights_file = fake_cache / "weights.pt"
        _save_random_weights(weights_file)

        with (
            patch("purepychess.weights._CACHE_DIR", fake_cache),
            patch("purepychess.weights._WEIGHTS_FILE", weights_file),
            patch("huggingface_hub.hf_hub_download") as mock_dl,
        ):
            from purepychess.weights import get_weights_path
            result = get_weights_path()

        mock_dl.assert_not_called()
        assert result == weights_file

    def test_raises_runtime_error_on_download_failure(self, tmp_path):
        """If weights absent and download fails, RuntimeError with instructions."""
        fake_cache    = tmp_path / "purepychess"
        weights_file  = fake_cache / "weights.pt"

        with (
            patch("purepychess.weights._CACHE_DIR", fake_cache),
            patch("purepychess.weights._WEIGHTS_FILE", weights_file),
            patch(
                "purepychess.weights.get_weights_path",
                side_effect=RuntimeError("Could not download weights"),
            ),
        ):
            from purepychess.weights import get_weights_path as gwp
            with pytest.raises(RuntimeError, match="Could not download weights"):
                gwp()

    def test_runtime_error_message_contains_cache_path(self, tmp_path):
        """The error message should include the local cache directory."""
        fake_cache   = tmp_path / "purepychess"
        weights_file = fake_cache / "weights.pt"

        import huggingface_hub
        with (
            patch("purepychess.weights._CACHE_DIR", fake_cache),
            patch("purepychess.weights._WEIGHTS_FILE", weights_file),
            patch.object(
                huggingface_hub,
                "hf_hub_download",
                side_effect=Exception("network error"),
            ),
        ):
            import importlib
            import purepychess.weights as wmod
            importlib.reload(wmod)

            with pytest.raises(RuntimeError) as exc_info:
                wmod.get_weights_path()

            assert str(fake_cache) in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_weights
# ---------------------------------------------------------------------------

class TestLoadWeights:
    def test_loads_state_dict_correctly(self, tmp_path):
        """load_weights fills the model with saved parameters."""
        fake_cache   = tmp_path / "purepychess"
        fake_cache.mkdir()
        weights_file = fake_cache / "weights.pt"

        # Save specific weights so we can verify they loaded
        model_a = ChessNet()
        # Give model_a a distinctive parameter value
        with torch.no_grad():
            for p in model_a.parameters():
                p.fill_(0.123)
        torch.save(model_a.state_dict(), weights_file)

        model_b = ChessNet()
        with (
            patch("purepychess.weights._CACHE_DIR", fake_cache),
            patch("purepychess.weights._WEIGHTS_FILE", weights_file),
        ):
            import purepychess.weights as wmod
            import importlib
            importlib.reload(wmod)
            wmod.load_weights(model_b, device=torch.device("cpu"))

        # model_b should now have all-0.123 weights
        for p in model_b.parameters():
            assert torch.allclose(p, torch.full_like(p, 0.123))

    def test_load_weights_returns_model(self, tmp_path):
        """load_weights should return the model for chaining."""
        fake_cache   = tmp_path / "purepychess"
        fake_cache.mkdir()
        weights_file = fake_cache / "weights.pt"
        _save_random_weights(weights_file)

        model = ChessNet()
        with (
            patch("purepychess.weights._CACHE_DIR", fake_cache),
            patch("purepychess.weights._WEIGHTS_FILE", weights_file),
        ):
            import purepychess.weights as wmod
            import importlib
            importlib.reload(wmod)
            result = wmod.load_weights(model, device=torch.device("cpu"))

        assert result is model


# ---------------------------------------------------------------------------
# download_weights
# ---------------------------------------------------------------------------

class TestDownloadWeights:
    def test_download_weights_delegates_to_get_weights_path(self, tmp_path):
        """download_weights() must call get_weights_path()."""
        fake_cache   = tmp_path / "purepychess"
        fake_cache.mkdir()
        weights_file = fake_cache / "weights.pt"
        _save_random_weights(weights_file)

        with (
            patch("purepychess.weights._CACHE_DIR", fake_cache),
            patch("purepychess.weights._WEIGHTS_FILE", weights_file),
        ):
            import purepychess.weights as wmod
            import importlib
            importlib.reload(wmod)

            with patch.object(wmod, "get_weights_path", return_value=weights_file) as mock_gwp:
                wmod.download_weights()
                mock_gwp.assert_called_once()
