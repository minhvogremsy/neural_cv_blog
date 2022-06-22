# Проект трекера TrackerDaSiamRPN действия которого основаны на нейронных сетях

Для корректной работы необходима версия cv2 не ниже 4.5.3 с поддержкой CUDA. 
Время вывода: 
|Среда выполнения|xavier        				 |PC (2070s)         		   | NANO		|
|----------------|-------------------------------|-----------------------------|------------|
|CUDNN			 | 20ms Перезахват 80ms	                 |10-12ms           			 |35-37ms (50-60ms 5W); MEM: 36% (1400M); CPU: 73%	|
|Opencv          |`25-30ms Перезахват 90ms        			 |7-22ms            			|75-79ms (108-120ms 5W); MEM: 30% (1200M); CPU: 53%	|



Для сборки есть 2 cmake файла. 1 для WIN второй для linux - в WIN файле захардкожены пути и для корректной работы библиотеки нужно перенести в папку C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64 используемые библиотеки (из C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib и ФАЙЛЫ tensorrt) 

Далее mkdir build
cd build 
cmake ..
cmake --build .
export DISPLAY=:0
./trt_efficientdet

В файле tracker_dasiamrpn.cpp есть 2 препроцессора - OPENCV и CUDNN  (влияет на то через что будет выполнен вывод)
> Нужно установить pycuda для этого нужно выполнить:
- pip3 install -U setuptools
- pip3 install -U pip
- pip3 install -U protobuf
- pip3 install numpy==1.19.3
- pip3 install Cython
- sudo apt-get update
- sudo apt-get install -y build-essential libatlas-base-dev
- sudo apt-get install -y libatlas-base-dev gfortran
- export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
- export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64\
         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
- pip3 install pycuda
- (подождать 10-15 минут, могут быть ошибки но нужно ждать)
- pip3 install Pillow
#### Для выполнения преобразования модели нужно выполнить :
- в файле  build_engine.py поменять строку self.config.max_workspace_size = 8 * (2 ** 30)  # 8 GB (поменять 8 на объем памяти который есть в системе, например 2 лучше 1 так как памяти кушает больше чем 1 при трансформации)
- в файле  build_engine.py установить (192 строка примерно) - > self.config.set_flag(trt.BuilderFlag.REFIT)
- python3 build_engine.py --onnx dasiamrpn_model_271.onnx --engine dasiamrpn_model32_r_271.trt --precision fp16