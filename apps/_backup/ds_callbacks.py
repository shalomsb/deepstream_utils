import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
from constants import PGIE_CLASS_ID_VEHICLE, PGIE_CLASS_ID_PERSON, PGIE_CLASS_ID_BICYCLE, PGIE_CLASS_ID_ROADSIGN


def osd_probe_test1(pad, info, u_data):
    """Probe for test1: object counts with colored bounding boxes."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0,
        }
        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.8)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = (
            f"Frame Number={frame_number} Number of Objects={num_rects} "
            f"Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]} "
            f"Person_count={obj_counter[PGIE_CLASS_ID_PERSON]}"
        )
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def osd_probe_test2(pad, info, u_data):
    """Probe for test2: object counts + tracker past frame metadata."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0,
        }
        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = (
            f"Frame Number={frame_number} Number of Objects={num_rects} "
            f"Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]} "
            f"Person_count={obj_counter[PGIE_CLASS_ID_PERSON]}"
        )
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    # Tracker past frame metadata
    l_user = batch_meta.batch_user_meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META:
            try:
                pPastDataBatch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break
            for miscDataStream in pyds.NvDsTargetMiscDataBatch.list(pPastDataBatch):
                print("streamId=", miscDataStream.streamID)
                print("surfaceStreamID=", miscDataStream.surfaceStreamID)
                for miscDataObj in pyds.NvDsTargetMiscDataStream.list(miscDataStream):
                    print("numobj=", miscDataObj.numObj)
                    print("uniqueId=", miscDataObj.uniqueId)
                    print("classId=", miscDataObj.classId)
                    print("objLabel=", miscDataObj.objLabel)
                    for miscDataFrame in pyds.NvDsTargetMiscDataObject.list(miscDataObj):
                        print('frameNum:', miscDataFrame.frameNum)
                        print('tBbox.left:', miscDataFrame.tBbox.left)
                        print('tBbox.width:', miscDataFrame.tBbox.width)
                        print('tBbox.top:', miscDataFrame.tBbox.top)
                        print('tBbox.height:', miscDataFrame.tBbox.height)
                        print('confidence:', miscDataFrame.confidence)
                        print('age:', miscDataFrame.age)
        try:
            l_user = l_user.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def pgie_src_probe_test3(pad, info, u_data):
    """Probe for test3: multi-stream object counts with FPS tracking.

    u_data is a dict with keys: 'perf_data', 'silent', 'measure_latency'.
    """
    perf_data = u_data.get("perf_data")
    silent = u_data.get("silent", False)
    measure_latency = u_data.get("measure_latency", False)

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer")
        return

    if measure_latency:
        num_sources = pyds.nvds_measure_buffer_latency(hash(gst_buffer))
        if num_sources == 0:
            print("Unable to get number of sources for latency measurement")

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0,
        }

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        if not silent:
            print(f"Frame Number={frame_number} Number of Objects={num_rects} "
                  f"Vehicle_count={obj_counter[PGIE_CLASS_ID_VEHICLE]} "
                  f"Person_count={obj_counter[PGIE_CLASS_ID_PERSON]}")

        stream_index = f"stream{frame_meta.pad_index}"
        perf_data.update_fps(stream_index)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
