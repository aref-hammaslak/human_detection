.venv/bin/python3 -m grpc_tools.protoc \
    --python_out=src/proto_bufs \
    --grpc_python_out=src/proto_bufs \
    --proto_path=src/protos human_detection.proto