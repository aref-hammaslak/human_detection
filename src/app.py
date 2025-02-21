from detect import Dectect
import os

def main():
    onnx_model_path = os.path.abspath("humman_detector.onnx")
    iou = 0.45
    cof = 0.5
    img_size = 640
    detect = Dectect(onnx_model_path, iou, cof, img_size)
    detect.predict_from_webcam()
    
if __name__ == "__main__":
    main()