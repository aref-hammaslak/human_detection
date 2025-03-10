import queue
import time
import cv2
import threading
import time
from  ultralytics import YOLO
import numpy as np
import os


class FrameCaptureThread(threading.Thread):
    def __init__(self, source, queue, ):
        super().__init__()
        self.source = source
        self.queue = queue
        self.running = True
        self.imgsz = 640
        self.cap = cv2.VideoCapture(self.source )

    def preprocess(self, img):
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        return img

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # print("Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue
            
            if frame is None or frame.size == 0:
                continue  # Skip invalid frames

            frame = self.preprocess(frame)


            # Put frame in queue (Drop old frames if queue is full)
            if not self.queue.full():
                self.queue.put(frame)
            else:
                self.queue.get()  # Remove old frame
                self.queue.put(frame)

    def stop(self):
        self.running = False
        self.cap.release()


class FrameProcessingThread(threading.Thread):
    def __init__(
        self,
        queue,
        model_path,
        iou=os.getenv('IOU', 0.45),
        conf=os.getenv('CONF', 0.4),
        target_classes=[0],
        fps_delay=60
        ):

        super().__init__()
        self.queue = queue
        self.running = True
        self.model = YOLO(model_path, task='detect')  # Load YOLO model from Ultralytics
        self.img_size = 640
        self.conf = conf
        self.target_classes = target_classes
        self.iou = iou
        self.conf = conf
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_delay = fps_delay
        
        print(f"fps will be printed after: {self.fps_delay}s") 

    def run(self):
        while self.running:
            if not self.queue.empty():
                frame = self.queue.get()

                if frame is None or frame.size == 0:
                    continue
                
                print(f"frame shape: {frame.shape}")

                results = self.model(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.target_classes,
                    verbose=False
                    )  # Perform detection
                
                output = results[0].boxes.xyxy
                print(f"( {len(output)} ) objects Detected")
                print(f"Bouding boxes: {output}")
                print("------------------------------------------------------")

                # Display the processed frame
                # cv2.imshow("Processed Frame", results[0].plot())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

                if self.start_time < time.time() - self.fps_delay:
                    # Calculate and display FPS
                    self.frame_count += 1
                    elapsed_time = time.time() - self.start_time -self.fps_delay
                    fps = self.frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")

    def stop(self):
        self.running = False
        cv2.destroyAllWindows()



class Detect:
    def __init__(self, source, model_path="yolo11.onnx"):
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = FrameCaptureThread(source, self.frame_queue)
        self.processing_thread = FrameProcessingThread(self.frame_queue, model_path= model_path)

    def start(self):
        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self):
        self.capture_thread.stop()
        self.processing_thread.stop()

