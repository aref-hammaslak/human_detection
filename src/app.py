from detect import Detect
import time
from dotenv import load_dotenv
import os
import cv2

# Load variables from .env file
load_dotenv()


def main():
    source = os.getenv('SOURCE', 0)# "rtsp://admin:AdminNasir58@192.168.1.107:554") # Webcam source
    print(f"Source: {source}")
    detect = Detect(source=source, model_path= os.getenv('MODEL_PATH','../model_openvino_model'))
    detect.start()

    try:
        while detect.capture_thread.running and detect.processing_thread.running:
            time.sleep(.1)

    except KeyboardInterrupt:
        print("Stopping threads...")
        detect.stop()


if __name__ == "__main__":
    main()