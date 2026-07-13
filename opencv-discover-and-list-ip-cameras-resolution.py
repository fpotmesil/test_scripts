import sys
import cv2
from onvif import ONVIFCamera
from wsdiscovery import WSDiscovery
import getpass

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

# Get all profiles from ONVIF camera
def get_profiles(host, port, user, password):
    try:
        cam = ONVIFCamera(host, port, user, password)
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        return media_service, profiles
    except Exception as e:
        print(f"❌ Failed to get profiles from {host}: {e}")
        return None, []

# Get RTSP URL for a specific profile
def get_rtsp_url(media_service, profile_token):
    try:
        stream_uri = media_service.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            },
            'ProfileToken': profile_token
        })
        return stream_uri.Uri
    except Exception as e:
        print(f"❌ Failed to get RTSP URL: {e}")
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

    # Get profiles
    media_service, profiles = get_profiles(host, port, username, password)
    if not profiles:
        sys.exit(1)

    print("\n🎥 Available Video Profiles:")
    for idx, profile in enumerate(profiles, start=1):
        try:
            res = profile.VideoEncoderConfiguration.Resolution
            enc = profile.VideoEncoderConfiguration.Encoding
            print(f"{idx}. {profile.Name} - {enc} {res.Width}x{res.Height}")
        except Exception:
            print(f"{idx}. {profile.Name} - (Resolution info unavailable)")

    # Let user choose profile
    try:
        profile_choice = int(input("\nSelect a profile number: "))
        if profile_choice < 1 or profile_choice > len(profiles):
            raise ValueError
    except ValueError:
        print("❌ Invalid profile selection.")
        sys.exit(1)

    selected_profile = profiles[profile_choice - 1]
    rtsp_url = get_rtsp_url(media_service, selected_profile.token)

    if rtsp_url:
        open_rtsp_stream(rtsp_url)

