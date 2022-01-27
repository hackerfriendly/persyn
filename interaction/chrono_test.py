'''
chronoperception: the passing of time from a bot's perspective
'''
import os
import uuid
import datetime as dt

from time import sleep

import pytest

from chrono import elapsed, get_cur_ts, natural_time

def test_elapsed_time():
    ''' Elapsed time '''
    assert elapsed("2022-01-27T10:54:31.000000-08:00", "2022-01-27T10:54:31.000000-08:00") == 0.0
    assert elapsed("2022-01-28T10:54:31.000000-08:00", "2022-01-29T10:54:31.100000-08:00") == 86400.1
    assert elapsed("2022-01-29T10:54:31.000000-08:00", "2022-01-28T10:54:31.100000-08:00") == 86399.9

    now = get_cur_ts()
    sleep(0.1)
    then = get_cur_ts()
    assert elapsed(now, then) == pytest.approx(0.1, abs=1e-3)

def test_natural_time():
    ''' Numbers to human estimates '''
    assert natural_time(1) == "late at night"
    assert natural_time(5) == "early morning"
    assert natural_time(9) == "morning"
    assert natural_time(13) == "afternoon"
    assert natural_time(17) == "evening"
    assert natural_time(21) == "night"

# def test_circadian():
