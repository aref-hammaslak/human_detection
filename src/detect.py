import onnx
import onnxruntime as ort
import cv2
from utils import Box, convert_boxes_to_xyxyc
import threading
from utils import FrameCaptureThread ,FrameProcessingThread
import queue
import time

# RTSP stream URL TEST IP CAMERA
RTSP_URL = "rtsp://admin:AdminNasir58@192.168.1.107:554"


class Dectect:
    def __init__(self, onnx_model_path, iou, cof, img_size):
        self.model = onnx.load(onnx_model_path)
        self.session = ort.InferenceSession(self.model.SerializeToString())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.iou_thr = iou
        self.cof_thr = cof
        self.img_size = img_size

        ################
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = FrameCaptureThread(RTSP_URL, self.frame_queue)
        self.processing_thread = FrameProcessingThread(self.frame_queue)
        
    def predict_from_webcam(self):
        cap = cv2.VideoCapture('rtsp://admin:AdminNasir58@192.168.1.107:554')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = self.preprocess(frame)
            input_data = {self.input_name: img}
            results = self.session.run([self.output_name], input_data)
            output = self.postprocess(results[0])[0]
            print(f"( {len(output)} ) objects Detected")
            print(f"Bouding boxes: {output}")
            print("------------------------------------------------------")
            cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def predict_from_webcam_with_threads(self):
        cap = cv2.VideoCapture('rtsp://admin:AdminNasir58@192.168.1.107:554')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = self.preprocess(frame)
            input_data = {self.input_name: img}
            results = self.session.run([self.output_name], input_data)
            output = self.postprocess(results[0])[0]
            print(f"( {len(output)} ) objects Detected")
            print(f"Bouding boxes: {output}")
            print("------------------------------------------------------")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()


    def predict_from_file(self, file_path):
        img = cv2.imread(file_path)
        img = self.preprocess(img)
        input_data = {self.input_name: img}
        results = self.session.run([self.output_name], input_data)
        output = self.postprocess(results[0])[0]
        print(f"( {len(output)} ) objects Detected")
        print(f"Bouding boxes: {output}")
        print("------------------------------------------------------")
        
    def preprocess(self, img):
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)
        img = img.reshape(1, 3, self.img_size, self.img_size)
        img = img.astype('float32')
        img = img / 255.0
        img = img.astype('float32')
        return img

    def postprocess(self, detections):
        batch_size, num_values, num_boxes = detections.shape
        processed_results = []
        for i in range(batch_size):
            boxes = [] 
            for j in range(num_boxes):
                x, y, width, height, confidence = detections[i, :, j]
                if confidence > self.cof_thr:  # Filter out low-confidence detections
                    boxes.append({
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height),
                        "confidence": float(confidence)
                    })
            final_boxes = self.non_max_suppression(boxes, iou_threshold=self.iou_thr)
            final_boxes = self.normalize(final_boxes)
            processed_results.append(final_boxes)

        return processed_results
    
    def normalize(self, boxes):
        """Normalize (x_center, y_center, width, height) to (0.0-1.0) range."""
        img_width, img_height = self.img_size, self.img_size
        normalized_boxes = []
        for box in boxes:
            x_center = box["x"] / img_width
            y_center = box["y"] / img_height
            width = box["width"] / img_width
            height = box["height"] / img_height
            normalized_boxes.append({
                "x": x_center,
                "y": y_center,
                "width": width,
                "height": height,
                "confidence": box["confidence"]
            })
        return normalized_boxes
    
    def iou(self, box1: Box, box2: Box) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes."""

        w1, h1 = box1["width"], box1["height"]
        w2, h2 = box2["width"], box2["height"]

        # Convert (x, y, width, height) to (x1, y1, x2, y2)
        box1 = convert_boxes_to_xyxyc([box1])
        box2 = convert_boxes_to_xyxyc([box2])
        x1_min, y1_min, x1_max, y1_max = box1["x1"], box1["y1"], box1["x2"], box1["y2"]
        x2_min, y2_min, x2_max, y2_max = box2["x1"], box2["y1"], box2["x2"], box2["y2"]

        # Compute intersection area
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Compute union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0
    
    def non_max_suppression(self, boxes, iou_threshold=0.5):
        """Apply NMS to remove overlapping boxes."""
        if len(boxes) == 0:
            return []

        sorted_boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        final_boxes = []

        while sorted_boxes:
            best_box = sorted_boxes.pop(0)  # Pick the highest-confidence box
            final_boxes.append(best_box)

            # Filter out boxes with high IoU
            sorted_boxes = [
                box for box in sorted_boxes
                if self.iou(best_box, box) < iou_threshold
            ]
            

        return final_boxes


# Capture Frames Thread

import os 

def main():

    onnx_model_path = os.path.abspath("yolo11m.onnx")
    iou = 0.45
    cof = 0.5
    img_size = 640
    detect = Dectect(onnx_model_path, iou, cof, img_size)
    detect.predict_from_webcam_with_threads()

    detect.capture_thread.start()
    detect.processing_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Stopping threads...")
        capture_thread.stop()
        processing_thread.stop()