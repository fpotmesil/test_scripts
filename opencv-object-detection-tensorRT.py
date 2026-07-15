import cv2
import torch
from ultralytics import YOLO
import os

#
# pip install ultralytics opencv-python
# Install PyTorch with CUDA (choose correct version for your GPU)
#
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Install TensorRT Python bindings (version must match your NVIDIA driver & CUDA)
#
# pip install nvidia-pyindex nvidia-pip nvidia-tensorrt
# 
# TensorRT must be installed on your system.
# On Windows, it comes with NVIDIA SDK Manager.
# On Linux, you can install via apt or tarball from NVIDIA.
# 
# Convert YOLOv8 to TensorRT:
# yolo export model=yolov8n.pt format=engine device=0 half=True
#
#
# Performance Notes
#   - FP16 mode (half=True) gives a big FPS boost with minimal accuracy loss.
#   - INT8 mode is even faster but requires calibration data.
#   - Lowering resolution (e.g., 640×480) can double FPS.
#   - On an RTX 3060, YOLOv8n TensorRT can exceed 150 FPS at 640×480.
#
# 
#

def main():
    # Check GPU availability
    if not torch.cuda.is_available():
        print("Error: CUDA GPU not detected. TensorRT requires NVIDIA GPU with CUDA.")
        return

    device = 'cuda'
    print(f"Using device: {device}")

    # Step 1: Export YOLOv8 model to TensorRT (only needs to be done once)
    trt_model_path = "yolov8n.engine"
    if not os.path.exists(trt_model_path):
        print("Exporting YOLOv8 model to TensorRT...")
        model = YOLO("yolov8n.pt")  # Nano model for speed
        model.export(format="engine", device=0, half=True)  # FP16 for speed
        print("TensorRT model exported.")

    # Step 2: Load TensorRT-optimized model
    print("Loading TensorRT model...")
    model = YOLO(trt_model_path)

    # Step 3: Open game camera feed
    cap = cv2.VideoCapture(0)  # Replace with your game camera index or RTSP URL
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Optional: Lower resolution for higher FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Step 4: Run TensorRT inference
        results = model(frame, verbose=False)

        # Step 5: Draw detections
        annotated_frame = results[0].plot()

        # Step 6: Display
        cv2.imshow("TensorRT YOLOv8 Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

