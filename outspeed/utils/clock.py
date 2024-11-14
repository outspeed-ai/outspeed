import time


class Clock:
    start__time = None

    @classmethod
    def start_playback(cls):
        if cls.start__time is None:
            cls.start__time = time.time()

    @classmethod
    def get_playback_time(cls) -> float:
        cls.start_playback()
        return time.time() - cls.start__time
