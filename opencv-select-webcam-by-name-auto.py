import cv2
import platform
import sys

#
# pip install opencv-python
# pip install pygrabber   # Windows only
# pip install pyudev      # Linux only
# 
#

def list_cameras_with_names():
    """
    Returns a list of (index, name) tuples for available cameras.
    Works differently depending on OS.
    """
    os_name = platform.system()

    if os_name == "Windows":
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError:
            print("Please install pygrabber: pip install pygrabber")
            sys.exit(1)

        graph = FilterGraph()
        devices = graph.get_input_devices()
        return [(i, name) for i, name in enumerate(devices)]

    elif os_name == "Linux":
        try:
            import pyudev
        except ImportError:
            print("Please install pyudev: pip install pyudev")
            sys.exit(1)

        context = pyudev.Context()
        devices = []
        index = 0
        for device in context.list_devices(subsystem='video4linux'):
            name = device.get('ID_V4L_PRODUCT', f"Camera {index}")
            devices.append((index, name))
            index += 1
        return devices

    elif os_name == "Darwin":  # macOS
        devices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append((i, f"Camera {i}"))
                cap.release()
        return devices

    else:
        print(f"Unsupported OS: {os_name}")
        return []

def open_camera_by_index(index):
    """
    Opens the camera with the given index and displays the feed.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if platform.system() == "Windows" else 0)

    if not cap.isOpened():
        print(f"Error: Cannot open camera index {index}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("webcam_output.mp4", fourcc, 20.0, frame_size)

    print(f"Opened camera index {index}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame.")
            break

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray scale", gray_scale)
        cv2.imshow(f"Camera {index}", frame)

        #
        # determine webcam size
        #
        ## print(frame.shape)
        out.write(frame)

        key = cv2.waitKey(1) ## & 0xFF

        if key == 27 or key == 113: ##ord('q'):
            print("Stopping recording and terminating")
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = list_cameras_with_names()
    if not cameras:
        print("No cameras found.")
        sys.exit(0)

    print("Available cameras:")
    for idx, name in cameras:
        print(f"{idx}: {name}")

    user_input = input("Enter camera index or part of name: ").strip()

    selected_index = None

    # Try to interpret as index
    if user_input.isdigit():
        idx = int(user_input)
        if any(c[0] == idx for c in cameras):
            selected_index = idx
        else:
            print(f"No camera found with index {idx}")
            sys.exit(0)
    else:
        # Search by partial name (case-insensitive)
        for idx, name in cameras:
            if user_input.lower() in name.lower():
                selected_index = idx
                print(f"Matched '{user_input}' to camera: {name} (index {idx})")
                break
        if selected_index is None:
            print(f"No camera found matching name '{user_input}'")
            sys.exit(0)

    open_camera_by_index(selected_index)

