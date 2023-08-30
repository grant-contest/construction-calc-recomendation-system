# Рекомендательная система проекта для стипендиального конкурса от РСХБ 2023
# Шаги для запуска проекта:
 
1. Установка всех зависимостей (из корня):
pip install -r requirements.txt

2. Генерация grpc-файлов для микросервиса (запуск из папки Microservice):
python -m grpc_tools.protoc -I. --python_out=proto --pyi_out=proto --grpc_python_out=proto step_recomendation.proto

3. Запуск сервера (из папки Microservice):
python server.py

4. Запуск клиента для тестирования сервера (из папки Microservice):
python server.py
