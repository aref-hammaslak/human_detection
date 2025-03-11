import queue
import time
import cv2
import threading
import time
from  ultralytics import YOLO
import numpy as np
import os


class FrameCaptureThread(threading.Thread):
    def __init__(self, source, queue, daemon=True):
        super().__init__()
        self.daemon = daemon
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
        fps_delay=60,
        daemon=True,
        parent=None,):

        super().__init__()
        self.daemon = daemon
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
        self.fps = 0
        self.parent : Detect = parent
        
        # print(f"fps will be printed after: {self.fps_delay}s") 

    def run(self):
        while self.running:
            if not self.queue.empty():
                frame = self.queue.get()

                if frame is None or frame.size == 0:
                    continue  # Skip invalid frames

                results = self.model(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    classes=self.target_classes,
                    verbose=False
                    )  # Perform detection
                
                self.frame_count += 1
                if self.frame_count == 10:
                    elapsed_time = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = result.names[int(box.cls[0])]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                cv2.imshow("Processed Frame", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.parent.stop()
                    break
                
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()



class Detect:
    def __init__(self, source, model_path="yolo11.onnx"):
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = FrameCaptureThread(source, self.frame_queue , daemon= True)
        self.processing_thread = FrameProcessingThread(self.frame_queue, model_path= model_path , daemon= True , parent= self)

    def start(self):
        self.capture_thread.start()
        self.processing_thread.start()

    def stop(self):
        self.capture_thread.stop()
        self.processing_thread.stop()

