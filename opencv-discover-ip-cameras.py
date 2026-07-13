import sys
import cv2
from onvif import ONVIFCamera
from wsdiscovery import WSDiscovery

# pip install opencv-python onvif_zeep ws-discovery
#
# How It Works
Discovery: Uses ws-discovery to find ONVIF devices on the LAN.
RTSP Retrieval: Uses onvif-zeep to query the camera’s media service for the RTSP URI.
Streaming: Opens the RTSP stream in OpenCV with cv2.VideoCapture.
Notes
You must know the camera’s username/password for RTSP access.
Some cameras require digest authentication — onvif-zeep handles this automatically.
If multiple cameras are found, you can modify the script to let you choose.
#

# Function to discover ONVIF cameras on the network
def discover_onvif_cameras(timeout=5):
    wsd = WSDiscovery()
    wsd.start()
    services = wsd.searchServices(timeout=timeout)
    wsd.stop()

    cameras = []
    for service in services:
        xaddrs = service.getXAddrs()
        for addr in xaddrs:
            if "onvif" in addr.lower():
                cameras.append(addr)
    return cameras

# Function to get RTSP URL from ONVIF camera
def get_rtsp_url(host, port, user, password):
    try:
        cam = ONVIFCamera(host, port, user, password)
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token
        stream_uri = media_service.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            },
            'ProfileToken': token
        })
        return stream_uri.Uri
    except Exception as e:
        print(f"❌ Failed to get RTSP URL from {host}: {e}")
        return None

# Function to open RTSP stream with OpenCV
def open_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Cannot open video stream.")
        sys.exit(1)

    print("✅ Streaming from:", rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break
        cv2.imshow("ONVIF Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🔍 Discovering ONVIF cameras...")
    cameras = discover_onvif_cameras()

    if not cameras:
        print("❌ No ONVIF cameras found.")
        sys.exit(1)

    print("✅ Found cameras:")
    for i, cam in enumerate(cameras):
        print(f"{i+1}. {cam}")

    # For simplicity, pick the first camera found
    first_cam_url = cameras[0]
    print(f"📡 Connecting to: {first_cam_url}")

    # Extract host and port from the URL
    try:
        host_port = first_cam_url.split("//")[1].split("/")[0]
        host, port = host_port.split(":")
        port = int(port)
    except ValueError:
        host = first_cam_url.split("//")[1].split("/")[0]
        port = 80  # default ONVIF port

    # Replace with your camera credentials
    USERNAME = "admin"
    PASSWORD = "12345"

    rtsp_url = get_rtsp_url(host, port, USERNAME, PASSWORD)
    if rtsp_url:
        open_rtsp_stream(rtsp_url)

