import cv2  # импорт модуля cv2
import argparse
import traceback
import numpy as np
import subprocess as sp

frame_width = 1920
frame_height = 1080
str_cale = 'scale=' + str(frame_width) + 'x' + str(frame_height)
url_link='udp://@172.29.161.53:8554'
command = ['ffmpeg',
                       # '-sync', 'ext',
                       '-i', url_link,
                       '-f', 'image2pipe',
                       '-c:v', 'h264_nvenc',
                       '-b:v', '2000k',
                       '-pix_fmt', 'bgr24',
                       '-nostats', '-hide_banner',
                       '-vf', str_cale,
                       '-loglevel', 'quiet',
                       '-vcodec', 'rawvideo', '-']

pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.STDOUT, bufsize=1000000)

while (True):
                
                raw_image = pipe.stdout.read(frame_width * frame_height * 3)
                frame = np.fromstring(raw_image, dtype='uint8')

                
                frame = frame.reshape((frame_height, frame_width, 3))

                cv2.imshow('image3', frame)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
                