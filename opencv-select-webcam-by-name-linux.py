import cv2
import pyudev

#
# pip install pyudev opencv-python 
#

def list_cameras_with_names():
    """
    Returns a list of (index, name) tuples for available cameras.
    """
    context = pyudev.Context()
    devices = []
    index = 0
    for device in context.list_devices(subsystem='video4linux'):
        name = device.get('ID_V4L_PRODUCT', 'Unknown')
        devices.append((index, name))
        index += 1
    return devices

def open_camera_by_index(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {index}")
        return

    print(f"Opened camera: {index}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = list_cameras_with_names()
    if not cameras:
        print("No cameras found.")
    else:
        print("Available cameras:")
        for idx, name in cameras:
            print(f"{idx}: {name}")

        try:
            choice = int(input("Enter the index of the camera to open: "))
            open_camera_by_index(choice)
        except ValueError:
            print("Invalid input.")

