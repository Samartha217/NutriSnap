"""Unit tests for utils/rate_limiter.py"""
import pytest
from unittest.mock import patch
from time import time


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear in-memory state between tests."""
    import utils.rate_limiter as rl
    rl._request_log.clear()
    yield
    rl._request_log.clear()


class TestCheckRateLimit:
    def test_first_request_allowed(self):
        from utils.rate_limiter import check_rate_limit
        assert check_rate_limit(chat_id=1) is True

    def test_requests_within_limit_allowed(self):
        from utils.rate_limiter import check_rate_limit
        with patch("utils.rate_limiter.RATE_LIMIT_RPM", 3):
            assert check_rate_limit(1) is True
            assert check_rate_limit(1) is True
            assert check_rate_limit(1) is True

    def test_request_over_limit_blocked(self):
        from utils.rate_limiter import check_rate_limit
        with patch("utils.rate_limiter.RATE_LIMIT_RPM", 3):
            check_rate_limit(1)
            check_rate_limit(1)
            check_rate_limit(1)
            assert check_rate_limit(1) is False  # 4th request blocked

    def test_different_users_independent(self):
        from utils.rate_limiter import check_rate_limit
        with patch("utils.rate_limiter.RATE_LIMIT_RPM", 1):
            check_rate_limit(user_a := 100)
            assert check_rate_limit(user_a) is False  # user A blocked
            assert check_rate_limit(user_b := 200) is True  # user B unaffected

    def test_old_requests_expire_from_window(self):
        from utils.rate_limiter import check_rate_limit, _request_log
        with patch("utils.rate_limiter.RATE_LIMIT_RPM", 2):
            # Inject old timestamps directly (outside the 60s window)
            _request_log[999] = [time() - 120, time() - 90]
            # Both are expired — new request should be allowed
            assert check_rate_limit(999) is True


class TestGetRemaining:
    def test_full_quota_remaining(self):
        from utils.rate_limiter import get_remaining
        with patch("utils.rate_limiter.RATE_LIMIT_RPM", 3):
            assert get_remaining(42) == 3

    def test_decrements_after_request(self):
        from utils.rate_limiter import check_rate_limit, get_remaining
        with patch("utils.rate_limiter.RATE_LIMIT_RPM", 3):
            check_rate_limit(42)
            assert get_remaining(42) == 2
