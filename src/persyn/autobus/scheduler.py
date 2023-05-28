import schedule
import datetime, logging

logger = logging.getLogger('scheduler')

# Catch and log exceptions as described in the Schedule docs:
# https://schedule.readthedocs.io/en/stable/exception-handling.html

class Scheduler(schedule.Scheduler):
    def _run_job(self, job):
        try:
            super()._run_job(job)
        except Exception as e:
            logger.exception(e)
            job.last_run = datetime.datetime.now()
            job._schedule_next_run()
