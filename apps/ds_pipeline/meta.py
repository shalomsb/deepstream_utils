"""Metadata iteration helpers for DeepStream pipelines.

Eliminates the verbose while/try/StopIteration boilerplate that appears
in every DeepStream probe function. Use generators to iterate over
frame metadata, object metadata, and user metadata with simple for-loops.
"""

import ctypes
import numpy as np
import pyds


def get_batch_meta(info):
    """Extract NvDsBatchMeta from a Gst.PadProbeInfo.

    Returns:
        (batch_meta, gst_buffer) or (None, None) if buffer is unavailable.
    """
    # gst_buffer is GstBuffer; it contains:
    # NvBufSurface - Actual pixel data (video frames) in GPU memory
    # NvDsBatchMeta - Metadata (e.g. object detection results) in CPU memory
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return None, None
    # batch_meta is a Python object wrapping the C struct NvDsBatchMeta, which is the metadata of the vudeo frame(s) in the buffer.
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    return batch_meta, gst_buffer


def iter_frames(batch_meta):
    """Yield each NvDsFrameMeta from a NvDsBatchMeta.

    Usage:
        for frame in iter_frames(batch_meta):
            print(frame.frame_num, frame.num_obj_meta)
    """
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            yield pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            return
        try:
            l_frame = l_frame.next
        except StopIteration:
            return


def iter_objects(frame_meta):
    """Yield each NvDsObjectMeta from a NvDsFrameMeta.

    Usage:
        for obj in iter_objects(frame_meta):
            print(obj.class_id, obj.confidence)
    """
    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            yield pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            return
        try:
            l_obj = l_obj.next
        except StopIteration:
            return


def iter_user_meta(meta_list, meta_type=None):
    """Yield NvDsUserMeta from a user_meta_list, optionally filtered by meta_type.

    Works with batch_meta.batch_user_meta_list, frame_meta.frame_user_meta_list,
    and obj_meta.obj_user_meta_list.

    Args:
        meta_list: The GList (e.g. frame_meta.frame_user_meta_list)
        meta_type: Optional pyds.NvDsMetaType to filter by

    Usage:
        for user_meta in iter_user_meta(frame_meta.frame_user_meta_list):
            ...
        for user_meta in iter_user_meta(batch_meta.batch_user_meta_list,
                                         pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META):
            ...
    """
    l_user = meta_list
    while l_user is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            return
        if meta_type is None or user_meta.base_meta.meta_type == meta_type:
            yield user_meta
        try:
            l_user = l_user.next
        except StopIteration:
            return


def iter_output_tensors(frame_meta, set_infer_done=True):
    """Yield NvDsInferTensorMeta from a frame's user metadata.

    Works with both nvinfer (output-tensor-meta=1) and
    nvinferserver (output_tensor_meta: true).

    When set_infer_done=True (default), automatically sets
    frame_meta.bInferDone = True only when tensors were produced.
    This ensures the tracker carries forward detections on frames
    skipped by interval > 0.

    Usage:
        for tensor_meta in iter_output_tensors(frame_meta):
            for i in range(tensor_meta.num_output_layers):
                layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                ...
    """
    found = False
    for user_meta in iter_user_meta(frame_meta.frame_user_meta_list,
                                     pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
        found = True
        yield pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
    if set_infer_done and found:
        frame_meta.bInferDone = True


def get_layer_data(tensor_meta, layer_idx, dtype=ctypes.c_float):
    """Extract numpy array from an inference output tensor layer.

    Args:
        tensor_meta: NvDsInferTensorMeta from iter_output_tensors()
        layer_idx: Output layer index (0-based)
        dtype: ctypes type (default c_float). Use c_int32 for integer outputs.

    Returns:
        numpy array with the layer's shape and data.
    """
    layer = pyds.get_nvds_LayerInfo(tensor_meta, layer_idx)
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(dtype))
    shape = tuple(layer.inferDims.d[i] for i in range(layer.inferDims.numDims))
    return np.ctypeslib.as_array(ptr, shape=shape)


def add_obj_meta(batch_meta, frame_meta, left, top, width, height,
                 class_id=0, confidence=1.0, unique_id=1,
                 border_color=(0.0, 1.0, 0.0, 0.8), border_width=3):
    """Create and attach an NvDsObjectMeta to a frame.

    Returns:
        NvDsObjectMeta — for setting additional fields if needed.
    """
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    obj_meta.unique_component_id = unique_id
    obj_meta.confidence = confidence
    obj_meta.class_id = class_id
    obj_meta.object_id = 0xffffffffffffffff

    rect = obj_meta.rect_params
    rect.left = left
    rect.top = top
    rect.width = width
    rect.height = height
    rect.border_width = border_width
    rect.border_color.set(*border_color)

    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
    return obj_meta
