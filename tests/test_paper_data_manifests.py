"""Manifest input validation."""

import pytest

from paper_data.manifests import track_inputs


def test_track_inputs_rejects_an_unknown_track() -> None:
    with pytest.raises(KeyError):
        track_inputs("nosuch")
