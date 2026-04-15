import yaml
from ds_pipeline import AppConfig

# Path where Triton Python models read their config
TRITON_CONFIG_PATH = "/triton/model_repo/gdino_config.yml"


class Config(AppConfig):
    def __init__(self, yaml_filename="config.yaml"):
        super().__init__(__file__, yaml_filename=yaml_filename)
        self.source = self.data["source"]
        self.tracker_config = self.resolve("tracker", "config_file")

        sm = self.data.get("streammux", {})
        self.streammux_width = sm.get("width", 1920)
        self.streammux_height = sm.get("height", 1080)
        self.streammux_batch_size = sm.get("batch_size", 1)

        pgie = self.data.get("pgie", {})
        self.interval = pgie.get("interval", 0)
        self.conf_threshold = pgie.get("conf_threshold", 0.3)
        self.nms_threshold = pgie.get("nms_threshold", 0.5)
        self.labels = pgie.get("labels", [])

        rtsp = self.data.get("rtsp", {})
        self.rtsp_port = rtsp.get("port", 8554)
        self.rtsp_udp_port = rtsp.get("udp_port", 5400)
        self.rtsp_mount = rtsp.get("mount", "/stream")
        self.rtsp_codec = rtsp.get("codec", "H264")
        self.rtsp_bitrate = rtsp.get("bitrate", 4000000)
        self.rtsp_enc_type = rtsp.get("enc_type", 0)

        # Generate configs from templates with streammux dimensions
        self.pgie_config = self._write_pgie_config()
        self._write_triton_config()

    def _write_pgie_config(self):
        """Generate nvinferserver config from template with streammux dimensions."""
        template_path = self.resolve("pgie", "config_file")
        generated_path = template_path.replace(".template", ".txt")

        with open(template_path, "r") as f:
            content = f.read()

        content = content.replace("{width}", str(self.streammux_width))
        content = content.replace("{height}", str(self.streammux_height))

        with open(generated_path, "w") as f:
            f.write(content)

        return generated_path

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
