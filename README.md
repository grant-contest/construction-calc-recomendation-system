# Steps to run project:

1. Generate protobuf files for python microservice: From Microservice folder run
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. step_recomendation.proto