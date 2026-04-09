import logging


class Logger(logging.Logger):

    def __init__(self, name, level=logging.DEBUG):
        super().__init__(name, level)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d_%m_%Y-%H_%M_%S",
        ))
        handler.setLevel(level)
        self.addHandler(handler)
