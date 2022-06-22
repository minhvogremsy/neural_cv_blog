### Задача - получить изображение с камеры через wifi - usb - hdmi.

##### Windows
- Качаем программу [https://gopro.my.salesforce.com/sfc/p/o0000000HJuF/a/3b000000cKtr/1oWO3n5pEXXDB_0pAAvPPkxphAO09FHPKtODP8S67_0] 
- В настройках (передача данных) переводим режим с MTP на GU-PRO webcam
- Запускаем трансляцию. 
 ***По умолчанию через USB udp://@172.29.161.53:8554***
#### Некоторые команды:

- http://172.29.161.51/gp/gpWebcam/START?res=1080 - запустить камеру с разрешением 1080
- http://172.29.161.51/gp/gpWebcam/STOP - остановить трансляцию 
- http:///172.29.161.51/gp/gpMediaList - получить список медия 

- Подробнее  можно посмотреть тут [https://github.com/KonradIT/goprowifihack/tree/master/HERO9]




##### Windows результат (установлен дополнительный софт)

|Стабилизация|ZOOM        				 |OpenCV        		                |
|----------------|-------------------------------|-----------------------------|
|Нет			 | 150ms         |300ms                        |
|Средняя          |300ms     |300ms            			|
|Максимальная     |-      |-            			|


##### linux 

##### FFMPEG


- ffmpeg -i udp://@172.29.161.53:8554?overrun_nonfatal=1 -c:v copy  -y -f segment -segment_time 10 "xxx-%03d.ts"   # минимально - рабочий пример
- ffmpeg -fflags nobuffer -flags low_delay -i udp://@172.29.161.53:8554?overrun_nonfatal=1 -c:v copy  -y -f segment -segment_time 10 "xxx-%03d.ts"  # так немного лучше
- ffmpeg -i udp://@172.29.161.53:8554?overrun_nonfatal=1 -vf "scale=1920x1080" -vcodec libx264 -preset ultrafast -tune zerolatency -b:v 1M -y -f segment -segment_time 60 "xxx-%03d.ts" # пример запускается но картина ужасная

##### Имеются 2 примера:
- FFMPEG_example.py - пример который показывает как можно обернуть FMPEG получить из него поток байт и вывести полученное изображение на экран - ***задержка более 10c***
- OPENCV_example.py - пример который показывает как выполнить вывод через OPENCV  - ***задержка более 10c***

- ***Замечано, что если вставить небольшую паузу в код (после или перед получением нового кадра) то картинка становится более ровная без потерянных кадров.***


##### https://github.com/KonradIT/gopro-py-api

- Пример удалось запустить, но не совсем понятно что с ним делать дальше. 
- Запуск примера - что то типа ***python3  GOPRO_example.py 172.29.161.53***


##### Полезные ссылки

- Справка по FMPEG [http://ffmpeg.org/ffmpeg-all.html#toc-udp]