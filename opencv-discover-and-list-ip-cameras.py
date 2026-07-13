import sys
import cv2
from onvif import ONVIFCamera
from wsdiscovery import WSDiscovery
import getpass

#
# pip install opencv-python onvif_zeep ws-discovery
#
# 
#

# Discover ONVIF cameras on the network
def discover_onvif_cameras(timeout=5):
    wsd = WSDiscovery()
    wsd.start()
    services = wsd.searchServices(timeout=timeout)
    wsd.stop()

    cameras = []
    for service in services:
        for addr in service.getXAddrs():
            if "onvif" in addr.lower():
                cameras.append(addr)
    return cameras

# Get RTSP URL from ONVIF camera
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

# Open RTSP stream with OpenCV
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

    print("\n✅ Found cameras:")
    for i, cam in enumerate(cameras, start=1):
        print(f"{i}. {cam}")

    # Let user choose a camera
    try:
        choice = int(input("\nSelect a camera number: "))
        if choice < 1 or choice > len(cameras):
            raise ValueError
    except ValueError:
        print("❌ Invalid selection.")
        sys.exit(1)

    selected_cam_url = cameras[choice - 1]
    print(f"📡 Selected: {selected_cam_url}")

    # Extract host and port
    try:
        host_port = selected_cam_url.split("//")[1].split("/")[0]
        if ":" in host_port:
            host, port = host_port.split(":")
            port = int(port)
        else:
            host = host_port
            port = 80
    except Exception:
        print("❌ Failed to parse camera address.")
        sys.exit(1)

    # Ask for credentials
    username = input("Username: ")
    password = getpass.getpass("Password: ")

    # Get RTSP URL and stream
    rtsp_url = get_rtsp_url(host, port, username, password)
    if rtsp_url:
        open_rtsp_stream(rtsp_url)

