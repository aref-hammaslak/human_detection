from pydantic import BaseModel
import cv2
import threading
import time
import onnx
import onnxruntime as ort

class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float

def convert_boxes_to_xyxyc(boxes):
    """Convert (x, y, width, height) to (x1, y1, x2, y2)"""
    xyxyc_boxes = []
    for box in boxes:
        x1 = box["x"] - box["width"] / 2
        y1 = box["y"] - box["height"] / 2
        x2 = box["x"] + box["width"] / 2
        y2 = box["y"] + box["height"] / 2
        xyxyc_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": box["confidence"]})
    
    
    return xyxyc_boxes[0] if len(xyxyc_boxes) == 1 else xyxyc_boxes


class FrameCaptureThread(threading.Thread):
    def __init__(self, rtsp_url, queue):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.queue = queue
        self.running = True
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame. Retrying...")
                time.sleep(0.1)
                continue

            # Resize frame (Optional: Improves processing speed)
            frame = cv2.resize(frame, (640, 480))

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
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True
        onnx_model_path = "yolo11m.onnx"
        self.model = onnx.load(onnx_model_path)
        self.session = ort.InferenceSession(self.model.SerializeToString())
        self.img_size = 640
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def run(self):
        while self.running:
            if not self.queue.empty():
                frame = self.queue.get()

                img = self.preprocess(frame)
                input_data = {self.input_name: img}
                results = self.session.run([self.output_name], input_data)
                output = self.postprocess(results[0])[0]
                print(f"( {len(output)} ) objects Detected")
                print(f"Bouding boxes: {output}")
                print("------------------------------------------------------")

                # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Display the processed frame
                cv2.imshow("Processed Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    break

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
                if confidence > .5:  # Filter out low-confidence detections
                    boxes.append({
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height),
                        "confidence": float(confidence)
                    })
            final_boxes = self.non_max_suppression(boxes, iou_threshold=0.45)
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
    
    def iou(self, box1, box2) -> float:
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


    def stop(self):
        self.running = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import queue

    RTSP_URL = "rtsp://admin:AdminNasir58@192.168.1.107:554"

    frame_queue = queue.Queue(maxsize=5)
    capture_thread = FrameCaptureThread(RTSP_URL, frame_queue)
    processing_thread = FrameProcessingThread(frame_queue)

    capture_thread.start()
    processing_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        print("Stopping threads...")
        capture_thread.stop()
        processing_thread.stop()