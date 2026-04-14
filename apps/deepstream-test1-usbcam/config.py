from ds_pipeline import AppConfig


class Config(AppConfig):
    def __init__(self):
        super().__init__(__file__)
        self.usbcam = self.data["usbcam"]
        self.pgie_config = self.resolve("pgie", "config_file")
