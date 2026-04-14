"""ds_pipeline — DeepStream Python toolkit.

Usage:
    from ds_pipeline import (
        create_pipeline, create_source_bin, create_streammux, create_pgie,
        run_pipeline, link_chain,
        get_batch_meta, iter_frames, iter_objects, iter_user_meta, iter_output_tensors,
    )
"""

# Element factories (public subset)
from ._elements import make_element
from ._elements import create_pipeline
from ._elements import create_streammux
from ._elements import create_pgie
from ._elements import create_sgie
from ._elements import create_tracker
from ._elements import create_pgie_inferserver
from ._elements import create_tiler
from ._elements import create_queue
from ._elements import create_fakesink
from ._elements import create_nvvidconv
from ._elements import create_nvosd
from ._elements import create_sink
from ._elements import create_capsfilter

# Bins
from .bins import create_source_bin
from .bins import create_rtsp_output_bin
from .bins import create_filesrc_bin

# Metadata (iteration + tensor extraction + object creation)
from .meta import get_batch_meta
from .meta import iter_frames
from .meta import iter_objects
from .meta import iter_user_meta
from .meta import iter_output_tensors
from .meta import get_layer_data
from .meta import add_obj_meta

# OSD convenience
from .osd import add_osd_text
from .osd import set_obj_label
from .osd import set_border_color
from .osd import count_objects

# Pipeline lifecycle
from .pipeline import run_pipeline
from .pipeline import link_chain

# RTSP
from .rtsp import start_rtsp_server

# Constants
from .constants import PGIE_CLASS_ID_VEHICLE
from .constants import PGIE_CLASS_ID_BICYCLE
from .constants import PGIE_CLASS_ID_PERSON
from .constants import PGIE_CLASS_ID_ROADSIGN
from .constants import MUXER_BATCH_TIMEOUT_USEC

# Configuration
from .config import parse_yaml
from .config import AppConfig

# Logger
from .logger import Logger
