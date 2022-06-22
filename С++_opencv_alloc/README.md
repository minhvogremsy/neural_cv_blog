#### В данном примере показано как при помощи Opencv и cuda ИСПОЛЬЗОВАТЬ память в изделиях типа jetson не выполняя операцуию копирования памяти в устройство. 
- Задача: передовать память по ссылке без копирования. 

Пример работы с изображением 4032х2688 пикселей на nano.

```
Frame size : 4032 x 2688, 10838016 Pixels 3 Channels
Resized Frame size : 2016 x 1344, 2709504 Pixels 3 Channels
Using standard memory transfer
STANDARD:SETUP: 1.2344e-05 s
STANDARD:UPLOAD: 0.587766
STANDARD:RESIZE: 0.088207
STANDARD:ACCESS: 0.0207339
STANDARD:TOTAL: 0.696911
Using unified memory
UNIFIED:SETUP: 0.0374215
UNIFIED:UPLOAD: 0.0319723
UNIFIED:RESIZE: 0.0645897
UNIFIED:ACCESS: 0.00584715
UNIFIED:TOTAL: 0.140059
Using pinned memory
PINNED:SETUP: 0.032814
PINNED:UPLOAD: 0.0244617
PINNED:RESIZE: 0.0853513
PINNED:ACCESS: 0.00865627
PINNED:TOTAL: 0.151493
```