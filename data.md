# Статьи с репозиториями (где код есть или планируется что он будет)

#### Шумоподавление и прочие улучшения картинки
- https://github.com/ckkelvinchan/BasicVSR_PlusPlus код и статья про шумоподавление (12.04.2022, ***КОД ЕСТЬ - 25.04.2022*** , визуально неплохой эффект от использования решения. Производительность (предположительно) около 5 FPS colab K80)
- https://github.com/megvii-research/NAFNet - шумоподавление. (12.04.2022, код есть, код ***хорошо оформлен с примерами***)
- https://github.com/yz-wang/UCL-Dehaze - статья https://arxiv.org/pdf/2205.01871.pdf - про то как удалять дымку из изображений. На ***05.05.2022*** кода нет. Визуально выглядит неплохо.
- https://github.com/WangLiwen1994/DLN (решение 2019г) - решение которое улучшает ночную съмку. На GPU колаба 1 кадр = 0,008с (с учетом размера 360 x 640). Используется сырая torch модель. На гугл диске файл с примером вывода DLN_13102021.ipynb Файл модели всего 2,7 мб. Пример видео [https://drive.google.com/file/d/1lTA2yTLf-8x6fBmxWi19_nLn0A8st2Po/view?usp=sharing]
- https://github.com/wybchd/DPENet - что то про удаление дождя с видеоизображений. Ессть код но он не полный (на 10.05.2022). ***Из полезного*** в статье есть сравнения аналогичных решений - которые если что можно посмотреть так же (если возникнет такая задача).
- https://arxiv.org/ftp/arxiv/papers/2205/2205.03464.pdf 10.05.2022. Статья про применение классических фильтров (в основном фильтров гауса) и некоторые вещи из классического анализа изображений.
- https://github.com/oneTaken/Awesome-Denoise - сборка ссылок на статьи и код по шумоподавлению (последняя публикация датированна 2020г). (25.25.2022)
- https://link.springer.com/content/pdf/10.1007/s40747-021-00428-4.pdf - статья на тему шумоподавления (25.25.2022). В целом говорится о том, что существуют специальные шумоподавляющие модели (например для медицинских изображений) и общие. Специальные как правило лучше.
- https://github.com/GitCVfb/CVR (***25.05.2022*** с кодом). Преобразования картинки сделанной с помощью скользящего затвора в изображение (видео) с глобальным затвором.
- https://github.com/House-Leo/RWSR-EDL - суперразрешение. На ***07.06.2022 кода нет, нет иснформации по поводу производительности ***

#### Моделирование шума (для оценки моделей по удалению шума)
- https://yorkucvil.github.io/Noise2NoiseFlow (***03.06.2022, кода нет***). Пример моделирования шума (в отличии от стандартного добавления шума типа соль и перец тут добавлеенный шум максимально близок к реальному)

#### Поиск замаскированных объектов 
- https://github.com/GewelsJI/DGNet  - ***26.05.2022*** с кодом. Позиционируется как самое быстрое и самое точное решение по поиску замаскированных объектов. 

#### Сопоставление ключевых точек
- https://github.com/cvg/Hierarchical-Localization/ (в репозитории есть пример ссылка на colab) - пример выполнения поиска ключевых точек. Гит репозиторий обновляемый. (05.05.2022). Пример стоит рассмотреть подробнее, так как в нем есть возможность определения положения искомого объекта -  направление взгляда)
- Название решений - SuperPoint, UnsuperPoint, R2D2, KP2D, D2-Net, DenserNet, LoFTR Пример в колабе "LoFTR_demo_single_pair 23082021 - поиск и сопоставление объектов.ipynb"
***SuperPoint, DISK or R2D2 упоминаються в слайде https://docs.google.com/presentation/d/1GEw_fDqmUhgUzNJRs_4MYb6Gi0K_z3Dc/edit#slide=id.p74 - работа по афинным преобразованием в рамках CVPR 2022***
***название задачи - local geometric correspondence*** 

- https://github.com/zju3dv/LoFTR (LoFTR) - пример который удалось запустить в колабе. Сама нейронная сеть состоит из нескольких подсетей, каждую из которых нужно переносить в отельности.
```sh
Были выявлены следующие отдельне нейронные структуры:
backbone  - успешно удалось перенести в onnx
loftr_coarse
fine_preprocess
loftr_fine
```
- https://github.com/luigifreda/pyslam/tree/master (pyslam) - ***20.06.2022*** в репозитории есть много примеров в том числе по задачи поиска особых точек.


#### Стабилизация видео
- [https://ci-gitlab-30.peleng.by/biaspaltsau_aa/biaspaltsau_aa/-/tree/6155_nano_dev/prj/vpi_stab_cv] различные примеры написанные мной в 7-9.2021 по стабилизации видео на С++. В этом репозитории в папке python_folder/Python_Stabilization_Video лежит базовый пример реализации стабилизации видео.

#### Обноружение оъектов и адаптация домена
- https://github.com/Vibashan/online-od код и статья про адаптацию домена (про то как детекторы при выходе за приделы обучающей выборки начинают хуже работать) (12.04.2022, ***кода пока нет***)
- https://github.com/RangiLyu/nanodet типа детектор который должен быть ***намного быстрее чем yolox (yolov4-tiny)*** 2021 (конец). На 25.05 реализована [https://ci-gitlab-30.peleng.by/6157/neural_detection/-/tree/master/%D0%A1%2B%2B_nanodet] версия на tensorrt. Результат по производительности выше чем у yolov4. Обучение - стандартное (заменить в config файле число классов, добавить свой данные в формате coco, изменить размер партии и число эпох (размер партии уменьшить в 2-4 раза что бы уместиться в 8 гб))
- https://github.com/obss/sahi смысл решения в том, что изображение бьется на области квадратной формы, обноружение идет в каждой конкретной области - скорость падает точность возрастает
- https://github.com/hustvl/YOLOP - YOLOP /
- https://github.com/HikariTJU/LD что то про дисциляцию при обноружении объектов. Дообучение нейронных сетей после осного цикла обучения. Типа их решение может дать +2% AP. 
- https://github.com/Duankaiwen/PyCenterNet PyCenterNet очередной 2 стадийный детектор по скорости сравнимый с yolov4 (Обычной) фишка которого заключатеся в том, что тут определяются не кординаты (нет якорей) а ключевые точки (верхий левый, нижний правый пиксель и центральный пиксель) есть часть с вниманием. Итоговая точность около Swin-Transforme. (19.04.2022, ***код есть, нет моделей***)
- https://github.com/naver-ai/vidt Очередной трансформер. Не для реального времени (ребята считают FPS на A100!)
- https://github.com/IDEACVR/MaskDINO (детекция + извлечение маски экземпляра 2 в 1 на разных выходах) - работа в первую очередь расчитана на точность а не на скорость. В тексте не смог найти скорость решения. *** на 07.06.2022 кода нет,есть базовый гит, стоит проверить позже *** Для поиска и выделения карты используються трансформеры.


#### Трекеры
- https://github.com/vision4robotics/TCTrack (2022) Быстрый трекер, но его не удалось завести в tensorrt (есть самописные слои со сложной логикой). На питоне возможна работа в реальном времени (xavier)
- https://github.com/litinglin/swintrack (2021 конец) трекер который основан на механизме внимания
- https://github.com/vision4robotics/HiFT (2021 конец) трекер который по проще чем TCTrack, основан на той же логике. Проблемы те же что и у TCTrack
- https://github.com/ifzhang/ByteTrack (2021) не сильно подходящий трекер из за ниской скорости работы
- https://github.com/RISC-NYUAD/SiamTPNTracker (2021) еще один трекер где есть вариант импорта в onnx, однако сама модель onnx не проходит конвертацию. Пуем удаления узла в котором была реализована динамическая инициализация весов (замена узла на аналогичный conv2D слой) реализована возможность конвертации трекера в tensorrt. Как удалять слой показано в примере onnx_demo_1_delete_graph.ipynb
- https://github.com/HYUNJS/SGT (***на 04.05.2022 кода нет***). Трекер покоторый показывает максимальную точность отслеживания на открытых наборах данных на 04.05.2022. По скорости ***SGT achieves MOTA of 76.7/76.3/72.8% while running at 23.0/23.0/19.9 FPS using a single V100 GPU in the MOT16/17/20 benchmarks, respectively*** - решение не для реального времени.
- https://github.com/vision4robotics/SiameseTracking4UAV (10/05/2022) + бумага. Репозиторий в котором выполнен обзор существующих сиамских трекеров (на момент публикации - 10.05.2022) и их сравнение с выполнением вывода на AGX Xavier (мы используем NX который медленнее). Из представленных решений наибольший интерес вызывает ***SiamRPN++_alex*** (как решение которое обладает максимальной производительностью при приемлемой точности). ***В репозитории есть ссылки на другие решения (гит репозитории)***
- https://github.com/fzh0917/SparseTT ***10.05.2022, кода нет***. Бумага про использование трансформеров при решении задачи трекинга. 

#### Сегментация 
- https://github.com/hustvl/TopFormer модель сегментации которая позиционирует себя как решение для мобильных устройств (код есть, 13.04.2022). Есть пример в python-folder. 
- https://github.com/open-mmlab/mmsegmentation - это фреймворк с очень большой подборкой моделей сегментации с примерами обучения. Быстрый просмотр не помг выявить модели которые можно применить в реальном времени на jetson устройствах. (13.04.2022). 
sh```
!git clone https://github.com/open-mmlab/mmsegmentation.git 
%cd mmsegmentation
!pip install -e .```
- https://arxiv.org/pdf/2205.01198.pdf - интересная (может быть) статья на тему поиска дефектов (трещин) в асфальте. Наибольший интерес прещдставляет сравнение различных нейронных структур на новом датасете. 
- https://arxiv.org/ftp/arxiv/papers/2206/2206.08605.pdf - ***20.06.2022*** большая статья в которой рассказывается про сравнение различных моделей для семантической сегментации в реальном времени. За базу (реального времени) был взят Nvidia Jetson Xavier AGX Developer Kit. Статья имеет относительно низкий порог входа и ***рекомендуется к прочтению***.

#### Сегментация и классификация
- HybridNets https://github.com/datvuthanh/HybridNets.git - суть работы заключается в том, что модель выполняет сразу несколько действий. Пример видео [https://www.youtube.com/watch?v=OdItqhKDnus]. Есть пример обучения, очень толстый фьючерэкстрактор - данное решение не для реального времени. Есть пример в колабе [https://colab.research.google.com/drive/1Uc1ZPoPeh-lAhPQ1CloiVUsOIRAVOGWA?usp=sharing] - в нем решение показывает скорость около 3 кадров с секунду. Так же решение довольно тяжело обучается. (29.04.2022)

#### Картографирование
- https://github.com/ualsg/GANmapper - пример как выполнять преобразование ч/б изображения (которое может быть получено?) в карту местности. Это пример не работает напрямую с цветным изображением. 

#### Поиск пути
- https://github.com/hku-mars/BALM (С++), ***20.06.2022***). 


#### Классификация
- https://github.com/ehuynh1106/TinyImageNet-Transformers (***24.05.2022***) - пример классификации через трансформеры (на 24.05.2022 самое свежее решение)
- https://github.com/snap-research/EfficientFormer (***важно 03.06.2022***) эта классификатор иображений (***на 03.06.2022 кода нет***) который значительно обошел imagenetv2 по точности при этом обладает такой же производительностью (заявлено около 1.5ms на iphone 12)

#### Разный код
- https://github.com/YuYang0901/LaViSE код и статья о том, как работаю глубокие сверточные сети
- https://jax.readthedocs.io/en/latest/index.html - решение (jax) которое позвоялет ускорить стандартные numpy операции блягодаря выполнению их на GPU или TPU (типа дополнительной обертки + возможность создовать компилируемые скрипты (под капотом))
- https://github.com/ARICKERT0003/GStreamer-Camera/tree/master Пример того как забирать камеру CSI через GSTREAMER и получать на выходе CV mat. Сложно понять в чем тут отличие например от базового сценария использования opencv. 
- https://gist.github.com/hum4n0id/cda96fb07a34300cdb2c0e314c14df0a большая справка по gstreamer

##### решения для поиска изменений в снимках дистанционного зондирования земли. 
- https://github.com/wgcban/SemiCD решение для доразметки данных на основе размечанных с высоким результатом (19.04.2022, ***код есть***)

##### Автономное вождение
- https://github.com/ApolloAuto/apollo (фреймворк для автономного вождения)
- https://arxiv.org/pdf/2205.09743.pdf (бумага, ссылка на git есть но кода на ***24.05.2022*** нет) - пример решение задачи поиска разметки и поиска движущихся объектов. 
- https://github.com/opendr-eu/opendr Фреймворк по ИИ для тесной интеграции с ROS

##### Разметка данных. 
- https://www.youtube.com/watch?v=LQe7XplKfcE - видео в котором описывается 10 существующих инструментов для разметки данных. В настоящий момент мы должны использовать вот эту [https://docs.google.com/document/d/1Ex4YNsSDODFLV9eGywzw6saiUpwpgJyb2MkpAxfLXsc/edit#] инструкцию и инстурмент типа https://github.com/microsoft/VoTT, однако VoTT это устаревший и не обновляемый инструмент от microsoft. Последняя вресия вышла 03 Jun 2020, данный инструмент начинает плохо работать если число фотографий переваливает за 10.000.


# интересные статьи 

#### Долговременное отслеживание
- https://arxiv.org/pdf/2204.05280.pdf MONCE Tracking Metrics: a comprehensive quantitative
performance evaluation methodology for object tracking . Что то про долговременное отслеживание и методики.  (12.04.2022)
- https://arxiv.org/pdf/2204.07927.pdf (19.04.2022) - статья про трекеры с их сравнениями

#### Геолокация (определение местоположения)
- https://arxiv.org/pdf/2204.08381.pdf
- https://github.com/rpartsey/pointgoal-navigation что то про автономную навигацию в комнате без GPS. Основная суть работы сводится к поиску пути на зашумленных изображений. (***03.06.2022***)

#### Языковая навигация 
- https://arxiv.org/pdf/2206.08645.pdf (https://github.com/PatZhuang/LSA) на ***20.06.2022*** кода пока нет. Суть идеи - научить робота ореинтироваться в незнакомой среде с использованием языковых каманд и трансформеров. 

#### Поиск аномалий
- https://github.com/LiUzHiAn/hf2vad (***20.06.2022**, есть код и примеры для запуска). Пример реализации алгоритма поиска аномалий в видео ряде (под аномалиями понимается нестандартное поведение людей). Пример - дверь в которую все входят (аномалией будет если кто то будет выбегать из двери)
- https://arxiv.org/pdf/2206.08568.pdf (***20.06.222***, более новая статья но без кода). Статья может быть интересна так как в ней есть сравнение полученного решения с другими анологичными решениями.


#### Обноружение оъектов и адаптация домена
- https://habr.com/ru/post/558406/ интересная статья на хабре про то как ребята учили свой детектор и про особенности работы детектора в квантованном виде (прочитал)
- https://docs.google.com/spreadsheets/d/1w_kwO8yCzHGhWW_FbwELgKzJ9EN_I6EjObrcsUZpYTw/edit#gid=0 это результат работы по yolox
- https://arxiv.org/pdf/2205.01571.pdf статья про то, как Китайцы переносили yolov2 на чип и делали вывод в железе. У них вышел чим на 40нм техпроцессе который выводит yolov2 1280x720@30FPS и который потребляет в 8 раз меньше чем стандартное решение на GPU. (04.05.2022)

#### Вывод нейронных сетей
- https://habr.com/ru/company/recognitor/blog/524980/ Довольно примитивно написано но в целом годная статья 




# Разные примеры кода
- https://github.com/jetsonhacks/camera-caps пример приложения написанного на python (QT) от jetsonhacks в котором показано 1) как генерировать строку для открытия камеры 2) как управлять параметрами самой камеры (встроенными параметрами - типа баланс белого и т п)
- https://github.com/jetsonhacks/gst-explorer пример приложения написанного на python QT в котором разбирается как посмотреть параметры из gst инспектора через GUI.



# Разные ссылки полученные (переданные) от/- Коревко
- https://aerotenna.readme.io/docs/what-is-ocpoc типа автопилот (но очень сырой и еще не вышел)
- https://github.com/PX4/PX4-Autopilot PX4-Autopilot
- https://developer.nvidia.com/embedded/jetson-orin
- https://www.votchallenge.net/vot2021/trackers.html - ссылка на VOT2021 (treackers)/ https://www.youtube.com/watch?v=N9kN_eSIF9A сама конференция
- https://www.youtube.com/watch?v=N9kN_eSIF9A - конференция по встраиваемым системам 2022
- https://cvpr2022.thecvf.com/workshop-schedule семинар по компьютерному зрению 2022
#### Docker
- https://ci-gitlab-30.peleng.by/devops/support/-/wikis/artifactory/Nexus-Docker-Repository
- https://ci-artifactory-30.peleng.by/#browse/browse:docker-group
#### Docker-END
- https://habr.com/ru/post/486202/ (отличие alpine от ubuntu)
- https://github.com/QMonkey/wsl-tutorial/blob/master/README.wsl2.md (WSL)
- https://github.com/DefTruth/lite.ai.toolkit инструмент для исследования моделей (большой)
- https://zhuanlan.zhihu.com/p/450586647 (tflite C++ ?)
- https://github.com/PINTO0309/PINTO_model_zoo (tenserflow C++ очень много примеров и моделей + конвектор в openviono )
- https://github.com/iwatake2222/play_with_tflite (tflite C++ вывод есть nanodet)
- https://shop.variscite.com/product/evaluation-kit/var-som-mx8m-nano-evaluation-kits/ (какой то дашборд)
- https://www.variscite.com/product/system-on-module-som/cortex-a72/spear-mx8-nxp-imx8/#memory еще какой то дашборд
- https://www.leopardimaging.com/product/nvidia-jetson-cameras/nvidia-agx-xavier-camera-kits/li-xavier-kit-imx265-x/li-xavier-kit-imx265m12/ решение с 6 камерами
- https://docs.google.com/spreadsheets/d/1yK24-C_6KQAvyKPNljTASOcTcopBeOGxfun6EnXZEA0/htmlview ПК который мы заказали для нейронных сетей


 # Просто ссылки которые захотелось сохранить
 - https://github.com/NVIDIA/TensorRT/blob/main/demo/EfficientDet/notebooks/EfficientDet-TensorRT8.ipynb (пример преобразования на питоне TensorRT)
 - https://habr.com/ru/company/otus/blog/649087/ статья про то как портировать C++ в python
 - https://arxiv.org/pdf/2204.12534.pdf интересная статья про то, как кодирование/декодирование видео влияет на качество трекера/сегментации. У проекта есть код [https://github.com/KuntaiDu/AccMPEG]
- https://huggingface.co/ какой то очень интересный сервис по машинному обучению который стоит пощупать. 
- https://github.com/niconielsen32/ComputerVision - гит репозиторий с большим числом примеров по компьютерному зрению. 

# Исследования
### Tensorrt
- https://docs.google.com/document/d/1Eu40R-ogMjW7JEnVVA0GoU-ePCGld252jDavLamku3E/edit Tensorrt - общие заметки

# Примеры которые собрались для angus_camera
- https://github.com/maoxuli/argus_samples (пример собрался.) В примере есть важная информация по поводу того, что нуэен gcc - 6 версии. 




