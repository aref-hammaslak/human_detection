from detect import Detect
import time
from pathlib import Path

def main():
    source = 0  # Webcam source
    print(f"Source: {source}")
    detect = Detect(source=source, model_path= 'yolo11.onnx')
    detect.start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Stopping threads...")
        detect.stop()


if __name__ == "__main__":
    main()