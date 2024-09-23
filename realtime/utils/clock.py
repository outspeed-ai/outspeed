import time


class Clock:
    start_time = None

    @classmethod
    def start_playback(cls):
        if cls.start_time is None:
            cls.start_time = time.time()

    @classmethod
    def get_playback_time(cls) -> float:
        cls.start_playback()
        return time.time() - cls.start_time
