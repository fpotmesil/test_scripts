#!/usr/bin/env python3
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

#
# Depends on TensorRT and DeepStream
# 
# https://developer.nvidia.com/deepstream-sdk
#
# DeepStream Python bindings are provided via pyds.
# pip install pyds
import pyds

# 
# Convert YOLOv8 to TensorRT:
# yolo export model=yolov8n.pt format=engine device=0 half=True
#


# Initialize GStreamer
Gst.init(None)

# Path to your TensorRT YOLOv8 engine file
YOLO_ENGINE_PATH = "yolov8n.engine"

# GStreamer pipeline for DeepStream
# v4l2src = USB camera (replace with nvarguscamerasrc for Jetson CSI camera or uridecodebin for RTSP)
pipeline_str = f"""
v4l2src device=/dev/video0 ! \
video/x-raw, width=640, height=480, framerate=30/1 ! \
videoconvert ! \
nvvideoconvert ! \
video/x-raw(memory:NVMM), format=NV12 ! \
nvinfer config-file-path=cfg_yolo.txt ! \
nvdsosd ! \
nvegltransform ! \
nveglglessink
"""

def bus_call(bus, message, loop):
    """Handle GStreamer bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    return True

def main():
    loop = GLib.MainLoop()

    # Create pipeline
    pipeline = Gst.parse_launch(pipeline_str)

    # Add bus watch
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main())

#
# cfg_yolo.txt
#
# 
# 
# [property]
# gpu-id=0
# net-scale-factor=0.0039215697906911373
# model-engine-file=yolov8n.engine
# labelfile-path=labels.txt
# batch-size=1
# network-mode=2
# num-detected-classes=80
# interval=0
# gie-unique-id=1
# process-mode=1
# network-type=0
# cluster-mode=2
# maintain-aspect-ratio=1
# parse-bbox-func-name=NvDsInferParseCustomYolo
# custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
# 
# [class-attrs-all]
# threshold=0.25
# 

