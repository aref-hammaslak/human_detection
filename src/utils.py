from pydantic import BaseModel

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
    