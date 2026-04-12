import yaml
from ds_pipeline import AppConfig

# Path where Triton Python models read their config
TRITON_CONFIG_PATH = "/triton/model_repo/gdino_config.yml"


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
        self.conf_threshold = pgie.get("conf_threshold", 0.3)
        self.nms_threshold = pgie.get("nms_threshold", 0.5)
        self.labels = pgie.get("labels", [])

        # Write config for Triton models so they pick up the same
        # labels and thresholds defined here in config.yaml.
        self._write_triton_config()

    def _write_triton_config(self):
        """Write labels + thresholds to a YAML file the Triton
        preprocess and postprocess Python models read at init."""
        triton_cfg = {
            "labels": self.labels,
            "conf_threshold": self.conf_threshold,
            "nms_threshold": self.nms_threshold,
            "num_select": 300,
        }
        with open(TRITON_CONFIG_PATH, "w") as f:
            yaml.dump(triton_cfg, f, default_flow_style=False)
