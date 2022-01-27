''' chronoception: the perception of time '''

import datetime as dt

def elapsed(ts1, ts2):
    ''' Elapsed seconds between two timestamps (str in isoformat) '''
    return abs((dt.datetime.fromisoformat(ts2) - dt.datetime.fromisoformat(ts1)).total_seconds())

def get_cur_ts():
    ''' Return a properly formatted timestamp string '''
    return str(dt.datetime.now(dt.timezone.utc).astimezone().isoformat())

def natural_time(hour=dt.datetime.now().hour):
    ''' Natural time of the day '''
    day_times = ("late at night", "early morning", "morning", "afternoon", "evening", "night")
    return day_times[hour // 4]
