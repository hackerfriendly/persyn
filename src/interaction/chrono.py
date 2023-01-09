''' chronoception: the perception of time '''

import datetime as dt
import humanize

def elapsed(ts1, ts2):
    ''' Elapsed seconds between two timestamps (str in isoformat) '''
    return abs((dt.datetime.fromisoformat(ts2) - dt.datetime.fromisoformat(ts1)).total_seconds())

def get_cur_ts():
    ''' Return a properly formatted timestamp string '''
    return str(dt.datetime.now(dt.timezone.utc).astimezone().isoformat())

def ago(since):
    ''' Return a human friendly estimate of elapsed time since ts '''
    return humanize.naturaldelta(dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(since))

def natural_time(hour=None):
    ''' Natural time of the day '''
    if hour is None:
        hour = dt.datetime.now().hour
    day_times = ("late at night", "early morning", "morning", "afternoon", "evening", "night")
    return day_times[hour // 4]

def today():
    ''' Natural day of the year '''
    return dt.date.today().strftime(f"%A %B {humanize.ordinal(dt.date.today().day)}, %Y")
