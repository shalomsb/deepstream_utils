from ds_pipeline import AppConfig


class Config(AppConfig):
    def __init__(self, yaml_filename="config.yaml"):
        super().__init__(__file__, yaml_filename=yaml_filename)
        self.source = self.data["source"]
        self.pgie_config = self.resolve("pgie", "config_file")
        self.tracker_config = self.resolve("tracker", "config_file")
        sm = self.data.get("streammux", {})
        self.streammux_width = sm.get("width", 1920)
        self.streammux_height = sm.get("height", 1080)
        self.streammux_batch_size = sm.get("batch_size", 1)
        pgie = self.data.get("pgie", {})
        self.inference_width = pgie.get("inference_width", 1280)
        self.inference_height = pgie.get("inference_height", 720)
        self.labels_file = pgie.get("labels_file", "/models/yolo26/labels.txt")
