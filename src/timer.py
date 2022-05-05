import time


class Timer:
    """
    処理時間を測る
    startからstopまでにかかった時間を累算する
    """

    def __init__(self):
        self.__start_time = None
        self.__elapsed_time = 0

    def start(self):
        self.__start_time = time.time()

    def stop(self):
        self.__elapsed_time += time.time() - self.__start_time
        self.__start_time = None

    @property
    def elapsed_time(self):
        return self.__elapsed_time
