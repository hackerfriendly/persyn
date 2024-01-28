''' chronoception: the perception of time '''

import datetime as dt

from typing import Optional

import humanize

def elapsed(ts1: str, ts2: Optional[str] = None) -> float:
    ''' Elapsed seconds between two timestamps (str in isoformat) '''
    if ts2 is None:
        ts2 = str(dt.datetime.now(dt.timezone.utc))

    return abs((dt.datetime.fromisoformat(ts2) - dt.datetime.fromisoformat(ts1)).total_seconds())

def get_cur_ts(epoch: Optional[int] = None) -> str:
    ''' Return a properly formatted timestamp string. If epoch is provided, use that instead of now(). '''
    if epoch is None:
        return str(dt.datetime.now(dt.timezone.utc).astimezone().isoformat())

    return str(dt.datetime.fromtimestamp(epoch, dt.timezone.utc).astimezone().isoformat())

def hence(since) -> str:
    ''' Return a human friendly estimate of elapsed time since ts '''
    return humanize.naturaldelta(dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(since))

def exact_time() -> str:
    ''' Precise time of day '''
    return f"{dt.datetime.now().hour}:{dt.datetime.now().minute:02}"

def natural_time(hour: Optional[int] = None) -> str:
    ''' Natural time of the day '''
    if hour is None:
        hour = dt.datetime.now().hour
    day_times = ("late at night", "in the early morning", "in the morning", "in the afternoon", "in the evening", "at night")
    return day_times[hour // 4]

def today() -> str:
    ''' Natural day of the year '''
    return dt.date.today().strftime(f"%A %B {humanize.ordinal(dt.date.today().day)}, %Y")

def seconds_ago(sec: Optional[int] = 0) -> float:
    ''' Return epoch time from m seconds ago '''
    return dt.datetime.now(dt.timezone.utc).timestamp() - sec # type: ignore
