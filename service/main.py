from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger


from service.tasks import get_weather_data, predict_forest_fire


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(
        get_weather_data.send,
        CronTrigger.from_crontab('0 * * * *'),
    )
    scheduler.add_job(
        predict_forest_fire.send,
        CronTrigger.from_crontab('5 * * * *'),
    )
    try:
        scheduler.start()
    except KeyboardInterrupt:
        scheduler.shutdown()
