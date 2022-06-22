#### Тут примеры из gstreamer - а

- !gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=640, height=480, framerate=30/1, format=NV12' ! nvvidconv flip-method=2 ! nvv4l2h264enc insert-sps-pps=true bitrate=1600000 ! rtph264pay ! udpsink port=5000 host=0.0.0.0
- gst-launch-1.0.exe -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! decodebin  ! videoconvert ! autovideosink



- !gst-launch-1.0 -vvv videotestsrc ! autovideosink   - тестовый запуск

- !gst-launch-1.0 -vv videotestsrc ! x264enc ! flvmux ! filesink location=xyz.flv   - тестовая запись в ФАЙЛ

- !gst-launch-1.0 -vvv videotestsrc ! tee name=t t. ! queue ! x264enc ! mp4mux ! filesink location=xyz.mp4 -e t. ! queue leaky=1 ! autovideosink sync=false  - тестовая запись в файл и проигрывание на мониторе



---
### РАБОЧИЕ ПРИМЕРЫ
-  writer.open("appsrc !  \
     videoconvert ! video/x-raw, format=(string)I420 ! omxh264enc control-rate=2 bitrate=8000000 ! video/x-h264, stream-format=byte-stream ! \
     rtph264pay mtu=1500 ! udpsink host=0.0.0.0 port=5000 sync=false async=false", 0, 30, cv::Size(frame_width, frame_height), true); ***пример  VideoWriter на вход которого можно подовать cv::Mat для трансляции через OpenCV***
- gst-launch-1.0 -v udpsrc port=5000 ! "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! h264parse ! decodebin ! videoconvert ! autovideosink sync=false ***проиграть на хосте***
 - sudo systemctl restart nvargus-daemon - перезагрузка аргус демона (рекомендуется выполнять если нужно перезапустить камеру)
 - v4l2-ctl -d0 --list-formats-ext 
 - gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)28/1' ! omxh264enc control-rate=2 preset-level=2 bitrate=12000000 ! video/x-h264, stream-format=byte-stream ! rtph264pay mtu=1500 ! udpsink host=0.0.0.0 port=5000 sync=false async=false   ***пример где мы используем gst для стриминга видео***

  - gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)28/1' ! nvv4l2h264enc bitrate=8000000  preset-level=2 ! video/x-h264, stream-format=byte-stream ! rtph264pay mtu=1500 ! udpsink host=0.0.0.0 port=5000 sync=false async=false   ***пример где мы используем gst для стриминга видео***


  gst-launch-1.0 nvarguscamerasrc sensor_id=0 ! 'video/x-raw(memory:NVMM), width=3264, height=2464, format=(string)NV12, framerate=(fraction)21/1' ! nvv4l2h264enc bitrate=12000000 ! video/x-h264, stream-format=byte-stream ! rtph264pay mtu=1500 ! udpsink host=0.0.0.0 port=5000 sync=false async=false


  ### Инструменты для GIT 
  #### Meld 
 -  http://sourceforge.net/projects/meld-installer/

 ```sh
After installing it http://sourceforge.net/projects/meld-installer/
I had to tell git where it was:
git config --global merge.tool meld
git config --global diff.tool meld
git config --global mergetool.meld.path “C:\Program Files (x86)\Meld\meld\meld.exe”

And that seems to work.  Both merging and diffing with “git difftool” or “git mergetool”

 ```

 ```sh
 sudo apt-get install libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev libxi-dev libxrender-dev libxcb1-dev libxcb-glx0-dev libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync0-dev libxcb-xfixes0-dev libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev llibxcb-xinerama0-dev libxkbcommon-dev libxkbcommon-x11-dev
 ```