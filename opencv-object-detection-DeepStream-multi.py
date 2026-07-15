#!/usr/bin/env python3
#
#
# multi-camera DeepStream YOLOv8 pipeline in Python that can process 4+ game feeds in parallel with <10 ms latency on an NVIDIA GPU.
#
#
#This uses DeepStream’s nvstreammux to batch multiple camera streams into a single TensorRT inference pass, keeping everything in zero-copy GPU memory for maximum speed
#
#

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
# pip install pyds
import pyds

# Initialize GStreamer
Gst.init(None)

# List of camera sources (USB, RTSP, or file paths)
CAMERA_SOURCES = [
    "/dev/video0",  # USB cam 1
    "/dev/video1",  # USB cam 2
    "rtsp://192.168.1.10:554/stream1",  # RTSP cam 3
    "rtsp://192.168.1.11:554/stream1"   # RTSP cam 4
]

YOLO_CONFIG_FILE = "cfg_yolo.txt"  # DeepStream YOLOv8 config

def create_source_bin(index, uri):
    """Create a GStreamer source bin for each camera."""
    bin_name = f"source-bin-{index}"
    nbin = Gst.Bin.new(bin_name)

    if uri.startswith("/dev/video"):
        src = Gst.ElementFactory.make("v4l2src", f"usb-source-{index}")
        src.set_property("device", uri)
        caps = Gst.ElementFactory.make("capsfilter", f"caps-{index}")
        caps.set_property("caps", Gst.Caps.from_string("video/x-raw,framerate=30/1,width=640,height=480"))
        conv = Gst.ElementFactory.make("videoconvert", f"conv-{index}")
        nbin.add(src)
        nbin.add(caps)
        nbin.add(conv)
        src.link(caps)
        caps.link(conv)
        pad = conv.get_static_pad("src")
    else:
        src = Gst.ElementFactory.make("uridecodebin", f"uri-source-{index}")
        src.set_property("uri", uri)
        src.connect("pad-added", lambda src, pad: pad.link(nbin.get_static_pad("src")))
        nbin.add(src)
        pad = None

    ghost_pad = Gst.GhostPad.new("src", pad)
    nbin.add_pad(ghost_pad)
    return nbin

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
    pipeline = Gst.Pipeline()

    # Stream muxer (batches multiple inputs)
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", len(CAMERA_SOURCES))
    streammux.set_property("width", 640)
    streammux.set_property("height", 480)
    streammux.set_property("batched-push-timeout", 10000)  # 10ms
    pipeline.add(streammux)

    # Add camera sources
    for i, uri in enumerate(CAMERA_SOURCES):
        src_bin = create_source_bin(i, uri)
        pipeline.add(src_bin)
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = src_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # YOLOv8 TensorRT inference
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", YOLO_CONFIG_FILE)

    # On-screen display
    nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Video sink
    sink = Gst.ElementFactory.make("nveglglessink", "video-output")
    sink.set_property("sync", False)

    # Add elements to pipeline
    for elem in [pgie, nvdsosd, sink]:
        pipeline.add(elem)

    # Link elements
    streammux.link(pgie)
    pgie.link(nvdsosd)
    nvdsosd.link(sink)

    # Bus watch
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
# YOLOv8 DeepStream Config (cfg_yolo.txt)
#
# 
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-engine-file=yolov8n.engine
labelfile-path=labels.txt
batch-size=4
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
parse-bbox-func-name=NvDsInferParseCustomYolo
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so

[class-attrs-all]
threshold=0.25

# Install DeepStream SDK (Jetson or x86 with NVIDIA GPU)
# https://developer.nvidia.com/deepstream-sdk
#
# Export YOLOv8 to TensorRT:
# yolo export model=yolov8n.pt format=engine device=0 half=True
#
# Prepare labels.txt (COCO labels if using pretrained YOLOv8)
#
#


