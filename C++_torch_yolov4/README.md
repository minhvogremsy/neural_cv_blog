# Пример вывода yolov4 через С++
Скорость вывода:

- darcnet 43FPS 512*512 yolov4
- torch 22FPS 512*512 yolov4
- darcnet  ~270FPS 512*512 yolov4-tiny
- torch ~190FPS 512*512 yolov4-tiny


Пример запуска:
- mkdri build && cd build
- cmake .. && cmake --build . --config Release
- yolo_torch.exe yolov4.cfg yolov4.weights demo.mp4


В папку с exe файлом нужно положить yolov4.cfg yolov4.weights demo.mp4

Для запуска нужен OpenCV и [LibTorch](https://pytorch.org/get-started/locally/) 