import cv2
from goprocam import GoProCamera, constants
##########
import sys
import ffmpeg
##########

print("Run modprobe v4l2loopback device=1 video_nr=44 card_label=\"GoPro\" exclusive_caps=1")
input("Hit enter when done!")
gopro = GoProCamera.GoPro(ip_address=GoProCamera.GoPro.getWebcamIP(
    sys.argv[1]), camera=constants.gpcontrol, webcam_device=sys.argv[1])
gopro.webcamFOV(constants.Webcam.FOV.Wide)
gopro.startWebcam()
print('OK startWebcam')
udp_stream = "udp://{}:8554".format(GoProCamera.GoPro.getWebcamIP(sys.argv[1]))
stream = ffmpeg.input(udp_stream, vsync=2, fflags="nobuffer",
                      flags="low_delay", probesize=3072)
stream = ffmpeg.output(stream, "/dev/video44",
                      ar="44100", vcodec='rawvideo', pix_fmt="yuv420p", format="v4l2")


print('ffmpeg run')
ffmpeg.run(stream)
print('ffmpeg stop')