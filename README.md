# Рекомендательная система проекта для стипендиального конкурса от РСХБ 2023
# Шаги для запуска проекта:
 
1. Генерация grpc-файлов для микросервиса: запуск из папки Microservice
python -m grpc_tools.protoc -I. --python_out=grpc --pyi_out=grpc --grpc_python_out=grpc step_recomendation.proto

2. Запуск сервера из папки Microservice:
python server.py

3. Запуск клиента для тестирования сервера из папки Microservice:
python server.py
