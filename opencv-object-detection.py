import cv2
import torch
from ultralytics import YOLO

#
# pip install ultralytics opencv-python torch torchvision torchaudio
#
#
# https://github.com/ultralytics/assets/releases
#
# https://developer.nvidia.com/computer-vision-sdk
#

def main():
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a pre-trained YOLOv8 model (small version for speed)
    # You can replace 'yolov8n.pt' with 'yolov8s.pt' or custom-trained weights
    model = YOLO('yolov8n.pt').to(device)

    # Open camera feed (0 = default webcam, replace with your game camera index or RTSP/USB path)
    cap = cv2.VideoCapture(0)  # Change to your game camera source

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Run detection on the frame
        results = model(frame, verbose=False)

        # Draw results on the frame
        annotated_frame = results[0].plot()

        # Display the frame
        cv2.imshow("GPU Object Detection", annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

