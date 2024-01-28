'''
chronoperception: the passing of time from a bot's perspective
'''
import pytest
from freezegun import freeze_time

from time import sleep

import pytest

from src.persyn.interaction.chrono import elapsed, get_cur_ts, hence, exact_time, natural_time, today, seconds_ago


# Constants for tests
TEST_TIME = "2024-01-01T12:00:00-08:00"
TEST_EPOCH = 1704139200  # Corresponding epoch time for TEST_TIME

def test_elapsed():
    earlier = "2023-01-01T11:00:00+00:00"
    later = "2023-01-01T13:00:00+00:00"
    assert elapsed(earlier, later) == 7200  # 2 hours difference
    assert elapsed(later, earlier) == 7200  # Order should not matter

    assert elapsed("2022-01-27T10:54:31.000000-08:00", "2022-01-27T10:54:31.000000-08:00") == 0.0
    assert elapsed("2022-01-28T10:54:31.000000-08:00", "2022-01-28T10:54:31.010000-08:00") == 0.01
    assert elapsed("2022-01-28T10:54:31.000000-08:00", "2022-01-29T10:54:31.100000-08:00") == 86400.1
    assert elapsed("2022-01-29T10:54:31.000000-08:00", "2022-01-28T10:54:31.100000-08:00") == 86399.9
    assert elapsed("2022-01-28T10:54:31.100000-08:00", "2022-01-29T10:54:31.000000-08:00") == 86399.9

    now = get_cur_ts()
    sleep(0.1)
    then = get_cur_ts()
    assert elapsed(now, then) == pytest.approx(0.1, abs=1e-3)

@freeze_time(TEST_TIME)
def test_get_cur_ts():
    assert get_cur_ts() == TEST_TIME
    assert get_cur_ts(TEST_EPOCH) == TEST_TIME

@freeze_time(TEST_TIME)
def test_hence():
    earlier = "2024-01-01T11:00:00-08:00"
    assert "hour" in hence(earlier)  # Should return a string with "hour" since it's an hour ago

@freeze_time(TEST_TIME)
def test_exact_time():
    ''' in UTC '''
    assert exact_time() == "20:00"

@pytest.mark.parametrize("hour, expected", [
    (None, "at night"),
    (1, "late at night"),
    (5, "in the early morning"),
    (9, "in the morning"),
    (13, "in the afternoon"),
    (17, "in the evening"),
    (21, "at night"),
])

@freeze_time(TEST_TIME)
def test_natural_time(hour, expected):
    assert natural_time(hour) == expected

@freeze_time(TEST_TIME)
def test_today():
    assert today() == "Monday January 1st, 2024"

@freeze_time(TEST_TIME)
def test_seconds_ago():
    assert seconds_ago(3600) == TEST_EPOCH - 3600  # 1 hour ago
