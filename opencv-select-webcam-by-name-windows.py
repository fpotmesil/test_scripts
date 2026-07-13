import cv2
from pygrabber.dshow_graph import FilterGraph

# 
# for windows, use pygrabber
#
# pip install pygrabber opencv-python

def list_cameras_with_names():
    """
    Returns a list of (index, name) tuples for available cameras.
    """
    graph = FilterGraph()
    devices = graph.get_input_devices()
    return [(i, name) for i, name in enumerate(devices)]

def open_camera_by_index(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
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

