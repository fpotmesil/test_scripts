#!/usr/bin/env python3
#
# 
#extend the multi-camera DeepStream YOLOv8 pipeline so that:

Each camera’s detections are written to a separate log file.
Optionally send detections over a WebSocket for a real-time game analytics dashboard.
We’ll hook into DeepStream’s nvinfer metadata via pyds to extract bounding boxes, class IDs, and confidence scores.

Multi-Camera DeepStream YOLOv8 with Logging + WebSocket
#
#
import sys
import gi
import json
import datetime
import asyncio
import websockets
from threading import Thread

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

# Initialize GStreamer
Gst.init(None)

# Camera sources
CAMERA_SOURCES = [
    "/dev/video0",
    "/dev/video1",
    "rtsp://192.168.1.10:554/stream1",
    "rtsp://192.168.1.11:554/stream1"
]

YOLO_CONFIG_FILE = "cfg_yolo.txt"

# WebSocket server settings
WS_ENABLED = True
WS_PORT = 8765
ws_clients = set()

# Async WebSocket server
async def ws_handler(websocket, path):
    ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        ws_clients.remove(websocket)

def start_ws_server():
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    ws_server = websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    loop.run_until_complete(ws_server)
    loop.run_forever()

# Start WebSocket server in background
if WS_ENABLED:
    Thread(target=start_ws_server, daemon=True).start()
    print(f"WebSocket server running on ws://0.0.0.0:{WS_PORT}")

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

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Extract detection metadata from DeepStream and log/send it."""
    frame_number = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        cam_id = frame_meta.source_id
        timestamp = datetime.datetime.now().isoformat()
        detections = []

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            detections.append({
                "class_id": int(obj_meta.class_id),
                "confidence": float(obj_meta.confidence),
                "bbox": {
                    "left": float(obj_meta.rect_params.left),
                    "top": float(obj_meta.rect_params.top),
                    "width": float(obj_meta.rect_params.width),
                    "height": float(obj_meta.rect_params.height)
                }
            })
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Write to per-camera log file
        log_filename = f"camera_{cam_id}_detections.log"
        with open(log_filename, "a") as f:
            f.write(json.dumps({
                "timestamp": timestamp,
                "camera_id": cam_id,
                "detections": detections
            }) + "\n")

        # Send over WebSocket
        if WS_ENABLED and ws_clients:
            msg = json.dumps({
                "timestamp": timestamp,
                "camera_id": cam_id,
                "detections": detections
            })
            asyncio.run(send_ws_message(msg))

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

async def send_ws_message(message):
    """Send message to all connected WebSocket clients."""
    if ws_clients:
        await asyncio.gather(*(client.send(message) for client in ws_clients))

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
    pipeline = Gst.Pipeline()

    # Stream muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", len(CAMERA_SOURCES))
    streammux.set_property("width", 640)
    streammux.set_property("height", 480)
    streammux.set_property("batched-push-timeout", 10000)
    pipeline.add(streammux)

    # Add camera sources
    for i, uri in enumerate(CAMERA_SOURCES):
        src_bin = create_source_bin(i, uri)
        pipeline.add(src_bin)
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = src_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # YOLOv8 inference
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", YOLO_CONFIG_FILE)

    # OSD
    nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    # Sink
    sink = Gst.ElementFactory.make("nveglglessink", "video-output")
    sink.set_property("sync", False)

    for elem in [pgie, nvdsosd, sink]:
        pipeline.add(elem)

    streammux.link(pgie)
    pgie.link(nvdsosd)
    nvdsosd.link(sink)

    # Attach probe to extract metadata
    osd_sink



#
#  multi-camera DeepStream YOLOv8 pipeline with per-camera logging and optional WebSocket streaming, but with some meaningful improvements:

Better async WebSocket handling (no blocking asyncio.run inside probe).
Non-blocking file writes using a background thread queue.
Cleaner metadata extraction with error safety.
Configurable output directory for logs.
Multi-Camera DeepStream YOLOv8 with Async Logging + WebSocket
# 
#
#
# 
#!/usr/bin/env python3
import sys
import gi
import json
import datetime
import asyncio
import websockets
from threading import Thread
from queue import Queue
from pathlib import Path

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

# Initialize GStreamer
Gst.init(None)

# Camera sources
CAMERA_SOURCES = [
    "/dev/video0",
    "/dev/video1",
    "rtsp://192.168.1.10:554/stream1",
    "rtsp://192.168.1.11:554/stream1"
]

YOLO_CONFIG_FILE = "cfg_yolo.txt"
LOG_DIR = Path("camera_logs")
LOG_DIR.mkdir(exist_ok=True)

# WebSocket settings
WS_ENABLED = True
WS_PORT = 8765
ws_clients = set()

# Thread-safe queue for logging
log_queue = Queue()

# Async WebSocket server
async def ws_handler(websocket, path):
    ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        ws_clients.remove(websocket)

async def ws_broadcast(message):
    """Broadcast message to all connected WebSocket clients."""
    if ws_clients:
        await asyncio.gather(*(client.send(message) for client in ws_clients))

def start_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_server = websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    loop.run_until_complete(ws_server)
    loop.run_forever()

# Background log writer
def log_writer():
    while True:
        cam_id, data = log_queue.get()
        if cam_id is None:
            break
        log_file = LOG_DIR / f"camera_{cam_id}_detections.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

# Start background services
if WS_ENABLED:
    Thread(target=start_ws_server, daemon=True).start()
    print(f"WebSocket server running on ws://0.0.0.0:{WS_PORT}")
Thread(target=log_writer, daemon=True).start()

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

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Extract detection metadata from DeepStream and queue for logging/WebSocket."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        cam_id = frame_meta.source_id
        timestamp = datetime.datetime.now().isoformat()
        detections = []

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            detections.append({
                "class_id": int(obj_meta.class_id),
                "confidence": float(obj_meta.confidence),
                "bbox": {
                    "left": float(obj_meta.rect_params.left),
                    "top": float(obj_meta.rect_params.top),
                    "width": float(obj_meta.rect_params.width),
                    "height": float(obj_meta.rect_params.height)
                }
            })
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        # Queue for logging
        log_queue.put((cam_id, {
            "timestamp": timestamp,
            "camera_id": cam_id,
            "detections": detections
        }))

        # Queue for WebSocket broadcast
        if WS_ENABLED and ws_clients:
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.create_task,
                ws_broadcast(json.dumps({
                    "timestamp": timestamp,
                    "camera_id": cam_id,
                    "detections": detections
                }))
            )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

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
    pipeline = Gst.Pipeline()

    # Stream muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", len(CAMERA_SOURCES))
    streammux.set_property("width", 640)
    streammux.set_property("height", 480)
    streammux.set_property("batched-push-timeout", 10000)
    pipeline.add(streammux)

    # Add camera sources
    for i, uri in enumerate(CAMERA_SOURCES):
        src_bin = create_source_bin(i, uri)
        pipeline.add(src_bin)
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = src_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    # YOLOv8 inference
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", YOLO_CONFIG_FILE)

    # OSD
    nvdsosd = Gst.ElementFactory.make("nvdsosd", "


#
#
# Version 3:
#
multi-camera DeepStream YOLOv8 pipeline with per-camera logging and optional WebSocket streaming.

Key improvements over the last version:

Added graceful shutdown for background threads and WebSocket server.
Added configurable detection confidence threshold.
Added camera name mapping for more readable logs.
Improved error handling in metadata parsing.
Added JSON pretty-print option for logs.
Ensured non-blocking WebSocket sends with exception handling.
Multi-Camera DeepStream YOLOv8 with Enhanced Logging + WebSocket
Python

Copy code
#!/usr/bin/env python3
import sys
import gi
import json
import datetime
import asyncio
import websockets
from threading import Thread, Event
from queue import Queue, Empty
from pathlib import Path

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

# Initialize GStreamer
Gst.init(None)

# Camera sources and friendly names
CAMERA_SOURCES = [
    ("/dev/video0", "FrontCam"),
    ("/dev/video1", "RearCam"),
    ("rtsp://192.168.1.10:554/stream1", "SideCam1"),
    ("rtsp://192.168.1.11:554/stream1", "SideCam2")
]

YOLO_CONFIG_FILE = "cfg_yolo.txt"
LOG_DIR = Path("camera_logs")
LOG_DIR.mkdir(exist_ok=True)

# Detection settings
CONF_THRESHOLD = 0.25
PRETTY_LOGS = False

# WebSocket settings
WS_ENABLED = True
WS_PORT = 8765
ws_clients = set()

# Thread-safe queue for logging
log_queue = Queue()
stop_event = Event()

# Async WebSocket server
async def ws_handler(websocket, path):
    ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        ws_clients.remove(websocket)

async def ws_broadcast(message):
    """Broadcast message to all connected WebSocket clients."""
    if ws_clients:
        send_tasks = []
        for client in list(ws_clients):
            try:
                send_tasks.append(client.send(message))
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

def start_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ws_server = websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    loop.run_until_complete(ws_server)
    loop.run_forever()

# Background log writer
def log_writer():
    while not stop_event.is_set():
        try:
            cam_name, data = log_queue.get(timeout=0.5)
        except Empty:
            continue
        log_file = LOG_DIR / f"{cam_name}_detections.log"
        with open(log_file, "a") as f:
            f.write(json.dumps(data, indent=2 if PRETTY_LOGS else None) + "\n")

# Start background services
if WS_ENABLED:
    Thread(target=start_ws_server, daemon=True).start()
    print(f"[INFO] WebSocket server running on ws://0.0.0.0:{WS_PORT}")
Thread(target=log_writer, daemon=True).start()

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

def osd_sink_pad_buffer_probe(pad, info, u_data):
    """Extract detection metadata from DeepStream and queue for logging/WebSocket."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except Exception:
            break

        cam_id = frame_meta.source_id
        cam_name = CAMERA_SOURCES[cam_id][1]
        timestamp = datetime.datetime.now().isoformat()
        detections = []

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except Exception:
                break

            if obj_meta.confidence >= CONF_THRESHOLD:
                detections.append({
                    "class_id": int(obj_meta.class_id),
                    "confidence": round(float(obj_meta.confidence), 3),
                    "bbox": {
                        "left": round(float(obj_meta.rect_params.left), 2),
                        "top": round(float(obj_meta.rect_params.top), 2),
                        "width": round(float(obj_meta.rect_params.width), 2),
                        "height": round(float(obj_meta.rect_params.height), 2)
                    }
                })
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        record = {
            "timestamp": timestamp,
            "camera_id": cam_id,
            "camera_name": cam_name,
            "detections": detections
        }

        # Queue for logging
        log_queue.put((cam_name, record))

        # Queue for WebSocket broadcast
        if WS_ENABLED and ws_clients:
            asyncio.get_event_loop().call_soon_threadsafe(
                asyncio.create_task,
                ws_broadcast(json.dumps(record))
            )

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def bus_call(bus, message, loop):
    """Handle GStreamer bus messages."""
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[INFO] End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"[ERROR] {err}, {debug}")
        loop.quit()
    return True

def main():
    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline()

    # Stream muxer
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", len(CAMERA_SOURCES))
    stream



