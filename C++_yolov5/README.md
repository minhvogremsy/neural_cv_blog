### yolov5
> Базовый git репозиторий проекта [https://github.com/ultralytics/yolov5/releases] . На 28.04.2022 проект живой (постоянные обновления)


### пролверка на jetson nano 

|Модель          |opencv_cuda_fp16        				 | tensorrt_fp16		|
|----------------|-------------------------------|----------|
|       yolov5n       |         10 FPS                |   15 FPS     	|
|       yolov5s       |          5 FPS                |     7.5 FPS    |

> yolov5_opencv_C++ пример вывода через opencv
> yolov5_tensorrt пример вывода где дополнительно добавлен tensorrt. forward из opencv просто закоментирован для сравнения при необходимости результатов. 