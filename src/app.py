from detect import Detect
import time
from dotenv import load_dotenv
import os


# Load variables from .env file
load_dotenv()


def main():
    source = os.getenv('SOURCE', 0) # Webcam source
    print(f"Source: {source}")
    detect = Detect(source=source, model_path= os.getenv('MODEL_PATH','/app/model_openvino_model'))
    detect.start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Stopping threads...")
        detect.stop()


if __name__ == "__main__":
    main()