from detect import Dectect
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'proto_bufs'))
from concurrent import futures

import grpc 
from grpc import StatusCode
import traceback
import proto_bufs.human_detection_pb2_grpc as human_detection_pb2_grpc
import proto_bufs.human_detection_pb2 as human_detection_pb2

class HummanDetectionServicer(human_detection_pb2_grpc.HumanDetectionServicer):
    def __init__(self, onnx_model_path, iou, cof, img_size):
        self.detect = Dectect(onnx_model_path, iou, cof, img_size)
        
    def serialize_result(self, result):
        humans = []
        for obj in result:
            bbox = human_detection_pb2.BoundingBox(x_min=obj['x1'], y_min=obj['y1'], x_max=obj['x2'], y_max=obj['y2'])
            human = human_detection_pb2.Human(id='', bounding_box=bbox, confidence=obj['confidence'])
            humans.append(human)
        return humans

    def Detect(self, request, context):
        image = request.image_data
        try:
            result = self.detect.predict_from_buffer(image)
            humans = self.serialize_result(result)
            return human_detection_pb2.DetectionResponse(humans=humans)
        except ValueError as e:
            context.set_details(str(e))
            context.set_code(StatusCode.INVALID_ARGUMENT)
            return human_detection_pb2.DetectionResponse()
        except Exception as e:
            traceback.print_exc()
            context.set_details('Internal server error')
            context.set_code(StatusCode.INTERNAL)
            return human_detection_pb2.DetectionResponse()

    def StreamDetect(self, request_iterator, context):
        for request in request_iterator:
            image = request.image_data
            try:
                
                result = self.detect.predict_from_buffer(image)
                humans = self.serialize_result(result)
                yield human_detection_pb2.DetectionResponse(humans=humans)
            except ValueError as e:
                context.set_details(str(e))
                context.set_code(StatusCode.INVALID_ARGUMENT)
                yield human_detection_pb2.DetectionResponse()
            except Exception as e:
                traceback.print_exc()
                context.set_details('Internal server error')
                context.set_code(StatusCode.INTERNAL)
                yield human_detection_pb2.DetectionResponse()

def server():
    onnx_model_path = os.path.abspath("human_detector.onnx")
    iou = 0.45
    cof = 0.5
    img_size = 640

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    human_detection_pb2_grpc.add_HumanDetectionServicer_to_server(HummanDetectionServicer(onnx_model_path, iou, cof, img_size), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Server started at port 50051")
    server.wait_for_termination() 
    
if __name__ == "__main__":
    server()