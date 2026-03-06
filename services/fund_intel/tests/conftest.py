import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def mock_cache():
    with patch("shared.cache.get_cache") as mock:
        cache = MagicMock()
        cache.get.return_value = None
        mock.return_value = cache
        yield cache


@pytest.fixture
def mock_requests():
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        yield mock_get, mock_post
