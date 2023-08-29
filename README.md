# Steps to run project:

1. Generate protobuf files for python microservice: From Microservice folder run
python -m grpc_tools.protoc -I. --python_out=grpc --pyi_out=grpc --grpc_python_out=grpc step_recomendation.proto

2. Run server:
