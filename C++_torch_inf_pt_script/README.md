# Пример тестирования времени вывода млоделей torch
Мы не обрабатываем результат вывода. Мы проверяем только то, что мы получаем тензор и перемещаем тензор в CPU

Пример запуска:
- mkdri build && cd build
- cmake .. && cmake --build . --config Release
- CMake_Torch.exe model.pt demo.mp4


В папку с exe файлом нужно положить yolov4.cfg yolov4.weights demo.mp4

Для запуска нужен OpenCV и [LibTorch](https://pytorch.org/get-started/locally/) 