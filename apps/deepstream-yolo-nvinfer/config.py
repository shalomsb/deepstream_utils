from ds_pipeline import AppConfig


class Config(AppConfig):
    def __init__(self):
        super().__init__(__file__)
        self.source = self.data["source"]
        self.pgie_config = self.resolve("pgie", "config_file")
        self.tracker_config = self.resolve("tracker", "config_file")
        sm = self.data.get("streammux", {})
        self.streammux_width = sm.get("width", 1920)
        self.streammux_height = sm.get("height", 1080)
        self.streammux_batch_size = sm.get("batch_size", 1)
        pgie = self.data.get("pgie", {})
        self.network_width = pgie.get("network_width", 640)
        self.network_height = pgie.get("network_height", 640)
        self.conf_threshold = pgie.get("conf_threshold", 0.25)
        self.nms_threshold = pgie.get("nms_threshold", 0.45)
        self.labels_file = pgie.get("labels_file", "/models/yolo11x/labels.txt")
