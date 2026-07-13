import cv2

def list_available_cameras(max_tested=10):
    """
    Test camera indices from 0 to max_tested-1 and return the working ones.
    """
    available_cameras = []
    for index in range(max_tested):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW avoids long delays on Windows
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

def open_camera(camera_index=0):
    """
    Open a specific camera by index and display the video feed.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Cannot open camera with index {camera_index}")
        return

    print(f"Opened camera index {camera_index}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow(f"Camera {camera_index}", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Step 1: List available cameras
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found.")
    else:
        print("Available cameras:", cameras)

        # Step 2: Select a camera (change index as needed)
        selected_index = cameras[0]  # Example: pick the first available
        open_camera(selected_index)

