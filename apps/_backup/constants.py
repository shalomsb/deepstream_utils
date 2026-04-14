PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3

MUXER_BATCH_TIMEOUT_USEC = 33000

class Test1Config:
    VIDEO = "/streams/sample_720p.h264"
    USBCAM = "/dev/video0"
    PGIE = "dstest1_pgie_config.txt"


class Test2Config:
    VIDEO = "/streams/sample_720p.h264"
    PGIE = "dstest2_pgie_config.txt"
    SGIE1 = "dstest2_sgie1_config.txt"
    SGIE2 = "dstest2_sgie2_config.txt"
    TRACKER = "dstest2_tracker_config.txt"


class Test3Config:
    PGIE = "dstest3_pgie_config.txt"
    PGIE_CLASSES = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]
    TILED_OUTPUT_WIDTH = 1280
    TILED_OUTPUT_HEIGHT = 720
    # 0 - CPU, 1 - GPU
    OSD_PROCESS_MODE = 0
    OSD_DISPLAY_TEXT = 1


class Test1RtspConfig:
    VIDEO = "/streams/sample_720p.h264"
    PGIE = "dstest1_pgie_config.txt"
    RTSP_PORT = 8554
    UDP_PORT = 5400
    RTSP_MOUNT = "/ds-test"
