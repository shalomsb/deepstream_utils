from ds_pipeline import AppConfig


class Config(AppConfig):
    def __init__(self, yaml_filename="config.yaml"):
        super().__init__(__file__, yaml_filename=yaml_filename)
        sources = self.data.get("sources")
        if sources is None:
            sources = [self.data["source"]]
        self.sources = sources
        self.num_sources = len(sources)
        self.pgie_config = self.resolve("pgie", "config_file")
        self.tracker_config = self.resolve("tracker", "config_file")
        sm = self.data.get("streammux", {})
        self.streammux_width = sm.get("width", 1920)
        self.streammux_height = sm.get("height", 1080)
        self.streammux_batch_size = sm.get("batch_size", self.num_sources)
        tiler = self.data.get("tiler", {})
        self.tiler_width = tiler.get("width", self.streammux_width)
        self.tiler_height = tiler.get("height", self.streammux_height)
        pgie = self.data.get("pgie", {})
        self.network_width = pgie.get("network_width", 704)
        self.network_height = pgie.get("network_height", 704)
        self.conf_threshold = pgie.get("conf_threshold", 0.5)
        self.labels_file = pgie.get("labels_file", "/models/rfdetr/labels.txt")
        rtsp = self.data.get("rtsp", {})
        self.rtsp_port = rtsp.get("port", 8554)
        self.rtsp_udp_port = rtsp.get("udp_port", 5400)
        self.rtsp_mount = rtsp.get("mount", "/stream")
        self.rtsp_codec = rtsp.get("codec", "H264")
        self.rtsp_bitrate = rtsp.get("bitrate", 4000000)
        self.rtsp_enc_type = rtsp.get("enc_type", 0)
