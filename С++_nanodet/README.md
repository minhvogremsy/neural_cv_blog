###### ADD 22/04/2022

###### В данном репозитории реализован пример использования nanodet с tensorrt. nanodet как ONNX модель с использованием opencv/onnx рантайма дает очень низкую производительность и не рассматривается (opencv cuda с моделью nanodet-plus-m-1.5x_416 FP16 дает около 3 FPS на хавьере, в то время как tensorrt дает около 60FPS)


> В настоящий момент в main еще присутствует код для сравнения результата tensorrt и opencv.
> Для сборки ```sh mkdir build && cd build && cmake --build .``` 

> Сборка модели - ```sh python3 build_engine.py --onnx nanodet-plus-m-1.5x_416.onnx --engine nanodet-plus-m-1.5x_416.trt --precision fp16 ```

> Для замены модели нужно не только указать в коде вручную название модели но и заменить число выводов в файле inference_fp16.cpp (строка int sz_1[] = {1,2125,112}; - для nanodet-plus-m_320.trt и sz_1[] = {1,3598,112}; для nanodet-plus-m-1.5x_416.trt, - размер вектора 112 зависит от числа классов. 

|Модель          |xavier        				 | NANO		|
|----------------|-------------------------------|----------|
|nanodet-plus-m-1.5x_416.trt| 15ms (65FPS) вывод с постобработкой 19ms(50FPS)|         	|
|nanodet-plus-m_320.trt     | 10ms(90FPS) вывод с постобработкой 13 ms(75FPS)| 23ms(42FPS) вывод с постобработкой 30ms(32FPS)    |
|nanodet-plus-m-1.5x_320.trt     |                                               |  31ms(31FPS) вывод с постобработкой 40ms(25FPS)   |


> tensorrt - 8.2.1.8

> Результат тренировки : для nanodet-plus-m-1.5x_320:
```sh 
Epoch:200
mAP: 0.37947960308508155
AP_50: 0.7578379923275416
AP_75: 0.3291161329538321
AP_small: 0.24390637522283884
AP_m: 0.48074277876207233
AP_l: 0.5017225544450493
```
> для nanodet-plus-m-1.5x_320 на 200 эпохе не было достигнуто насыщение. batch уменьшен  
> workers_per_gpu: 5 batchsize_per_gpu: 24

> Результат тренировки : для nanodet-plus-m_320:
```sh 
Epoch:360
mAP: 0.46862401260652514
AP_50: 0.8399544437310749
AP_75: 0.45720426634344175
AP_small: 0.3294984698706501
AP_m: 0.5689294171207497
AP_l: 0.5646033107876026
```

#### Замечания:
> 1) Для обучения модели нужен существенный обьем видеопамяти. На 2070s 8gb удается енормально обучать только nanodet-plus-m_320, при этом все равно размер партии приходиться уменьшать. 


#### Инструкция по тренировке:
> git clone https://github.com/RangiLyu/nanodet.git 
> Выбрать в папке C:\Users\biaspaltsau_aa\python_folder\NANODET\nanodet\config конфигурационный файл для модели которую нужно обучить
> Для обучения нужны данные в формате COCO (для конвертации yolo to coco можно использовать  [https://github.com/Taeyoung96/Yolo-to-COCO-format-converter])
> python tools/train.py CONFIG_FILE_PATH 
> Выполнить экспорт весов в onnx (nanodet-train/export_onnx.ipynb)

#### Калибровка INT8!
```sh 
python3 build_engine.py --onnx nanodet-plus-m-320_2class.onnx --engine nanodet-plus-m-320_2class_int8.trt --precision int8 --calib_num_images 25000 --calib_input /home/andry/image_from_calibration_model/winter_summer/
```
> Для колибровки нужен только сет данных который будет поступать на вход. На ***05.05.2022*** не совсем понятно как это будет работать с несколькими входами. 
> Калибровка на Xavier выполняется за 30 минут. После калибровки появиться файл (в папке с моделью) calibration.cache - в дальнейшем его можно использовать для повторной калибровки изображения.

|Модель          |FP16 (Xavier)        				 | INT8	(Xavier) |
|----------------|-------------------------------|----------|
|nanodet-plus-m_320.trt| 15ms (65FPS) вывод с постобработкой 19ms(50FPS)|  8ms (124.122 FPS) вывод с постобработкой 11ms (90FPS)|
