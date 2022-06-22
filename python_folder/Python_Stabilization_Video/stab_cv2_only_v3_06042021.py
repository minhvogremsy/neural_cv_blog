########################
#np обертка над стандартной библиотекой для питона (написана на с работает значительно быстрее чем чистый питон)
#
#
import numpy as np
import cv2
from collections import deque
##########################



##########################


### В первой версии мы будем использовать только один детектор особых точек GFTT. ###########
### Для наала мы реализуем сам детектор особых точек
###
class GFTT:
    def __init__(self, maxCorners=0, qualityLevel=0.01, minDistance=1,
                 mask=None, blockSize=3, useHarrisDetector=False, k=0.04):
        self.maxCorners = maxCorners
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance
        self.mask = mask
        self.blockSize = blockSize
        self.useHarrisDetector = useHarrisDetector
        self.k = k

    def detect(self, img):
        cnrs = cv2.goodFeaturesToTrack(img, self.maxCorners, self.qualityLevel, self.minDistance,
                                       mask=self.mask, blockSize=self.blockSize,
                                       useHarrisDetector=self.useHarrisDetector, k=self.k)

        return corners_to_keypoints(cnrs)

def corners_to_keypoints(corners):
    """функция для извлечения углов из cv2.GoodFeaturesToTrack и возврата cv2.KeyPoints"""
    if corners is None:
        keypoints = []
    else:
        keypoints = [cv2.KeyPoint(kp[0][0], kp[0][1], 1) for kp in corners]

    return keypoints
######################################################################################
#перепишем функцию resize
#
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # инициализируем размеры изображения, размер которого нужно изменить и
    # получить размер изображения
    dim = None
    (h, w) = image.shape[:2]

    # если и ширина, и высота равны None, то возвращаем
    # исходное изображение
    if width is None and height is None:
        return image

    # проверяем, равна ли ширина None
    if width is None:

        # вычислить отношение высоты и построить
        # размеры
        r = height / float(h)
        dim = (int(w * r), height)
    # в противном случае высота равна None
    else:
        # вычисляем отношение ширины и строим
        # размеры
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
############################################################################################
#
#часть функционала которая отвечает за "бордюр" или то как будет выглядить изрображение после стабилизации
#

############################################################################################

def auto_border_start(min_corner_point, border_size):
    """Определите координаты верхнего правого угла для автоматической обрезки границ.p

    :param min_corner_point: extreme corner component either 'min_x' or 'min_y'
    :param border_size: min border_size determined by extreme_frame_corners in vidstab process
    :return: adjusted extreme corner for cropping
    """
    return math.floor(border_size - abs(min_corner_point))


def auto_border_length(frame_dim, extreme_corner, border_size):
    """Определение высоты / ширины автоматической обрезки границы

    :param frame_dim: height/width of frame to be auto border cropped (corresponds to extreme_corner)
    :param extreme_corner: extreme corner component either 'min_x' or 'min_y' (corresponds to frame_dim)
    :param border_size: min border_size determined by extreme_frame_corners in vidstab process
    :return: adjusted extreme corner for cropping
    """
    return math.ceil(frame_dim - (border_size - extreme_corner))


def auto_border_crop(frame, extreme_frame_corners, border_size):
    """Рамка кадрирования для автоматической границы

    :param frame: frame to be cropped
    :param extreme_frame_corners: extreme_frame_corners attribute of vidstab object
    :param border_size: min border_size determined by extreme_frame_corners in vidstab process
    :return: cropped frame determined by auto border process
    """
    if border_size == 0:
        return frame

    frame_h, frame_w = frame.shape[:2]

    x = auto_border_start(extreme_frame_corners['min_x'], border_size)
    y = auto_border_start(extreme_frame_corners['min_y'], border_size)

    w = auto_border_length(frame_w, extreme_frame_corners['max_x'], border_size)
    h = auto_border_length(frame_h, extreme_frame_corners['max_y'], border_size)

    return frame[y:h, x:w]


def functional_border_sizes(border_size):
    """Рассчитать размер границы, используемый в процессе, для определения размера границы, указанного пользователем

    If border_size is negative then a stand-in border size is used to allow better keypoint tracking (i think... ?);
    negative border is then applied at end.

    :param border_size: user supplied border size
    :return: (border_size, neg_border_size) tuple of functional border sizes

    >>> functional_border_sizes(100)
    (100, 0)
    >>> functional_border_sizes(-10)
    (100, 110)
    """
    if border_size < 0:
        neg_border_size = 100 + abs(border_size)
        border_size = 100
    else:
        neg_border_size = 0

    return border_size, neg_border_size
###########################################################################################################



def crop_frame(frame, border_options):
    """Обработка обрезки рамки для автоматического размера границы и отрицательного размера границы

    if auto_border is False and neg_border_size == 0 then frame is returned as is

    :param frame: frame to be cropped
    :param border_options: dictionary of border options including keys for:
        * 'border_size': functional border size determined by functional_border_sizes
        * 'neg_border_size': functional negative border size determined by functional_border_sizes
        * 'extreme_frame_corners': VidStab.extreme_frame_corners attribute
        * 'auto_border': VidStab.auto_border_flag attribute
    :return: cropped frame
    """
    if not border_options['auto_border_flag'] and border_options['neg_border_size'] == 0:
        return frame

    if border_options['auto_border_flag']:
        cropped_frame_image = auto_border_crop(frame.image,
                                               border_options['extreme_frame_corners'],
                                               border_options['border_size'])

    else:
        frame_h, frame_w = frame.image.shape[:2]
        cropped_frame_image = frame.image[
                              border_options['neg_border_size']:(frame_h - border_options['neg_border_size']),
                              border_options['neg_border_size']:(frame_w - border_options['neg_border_size'])
                              ]

    cropped_frame = Frame(cropped_frame_image, color_format=frame.color_format)

    return cropped_frame




def build_transformation_matrix(transform):
    """Преобразовать список преобразований в матрицу преобразований

    :param transform: transform list as [dx, dy, da]
    :return: transform matrix as 2d (2, 3) numpy array
    """
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix


def border_frame(frame, border_size, border_type):
    """Convenience wrapper of cv2.copyMakeBorder for how vidstab applies borders

    :param frame: frame to apply border to
    :param border_size: int border size in number of pixels
    :param border_type: one of the following ['black', 'reflect', 'replicate']
    :return: bordered version of frame with alpha layer for frame overlay options
    """
    border_modes = {'black': cv2.BORDER_CONSTANT,
                    'reflect': cv2.BORDER_REFLECT,
                    'replicate': cv2.BORDER_REPLICATE}
    border_mode = border_modes[border_type]

    bordered_frame_image = cv2.copyMakeBorder(frame.image,
                                              top=border_size,
                                              bottom=border_size,
                                              left=border_size,
                                              right=border_size,
                                              borderType=border_mode,
                                              value=[0, 0, 0])

    bordered_frame = Frame(bordered_frame_image, color_format=frame.color_format)

    alpha_bordered_frame = bordered_frame.bgra_image
    alpha_bordered_frame[:, :, 3] = 0
    h, w = frame.image.shape[:2]
    alpha_bordered_frame[border_size:border_size + h, border_size:border_size + w, 3] = 255

    return alpha_bordered_frame, border_mode


def match_keypoints(optical_flow, prev_kps):
    """Сопоставьте ключевые точки оптического потока

    :param optical_flow: output of cv2.calcOpticalFlowPyrLK
    :param prev_kps: keypoints that were passed to cv2.calcOpticalFlowPyrLK to create optical_flow
    :return: tuple of (cur_matched_kp, prev_matched_kp)
    """
    cur_kps, status, err = optical_flow

    # storage for keypoints with status 1
    prev_matched_kp = []
    cur_matched_kp = []

    if status is None:
        return cur_matched_kp, prev_matched_kp

    for i, matched in enumerate(status):
        # store coords of keypoints that appear in both
        if matched:
            prev_matched_kp.append(prev_kps[i])
            cur_matched_kp.append(cur_kps[i])

    return cur_matched_kp, prev_matched_kp


def estimate_partial_transform(matched_keypoints):
    """

    :param matched_keypoints: output of match_keypoints util function; tuple of (cur_matched_kp, prev_matched_kp)
    :return: transform as list of [dx, dy, da]
    """
    cur_matched_kp, prev_matched_kp = matched_keypoints

    transform = cv2.estimateAffinePartial2D(np.array(prev_matched_kp),
                                           np.array(cur_matched_kp))[0]
    if transform is not None:
        # translation x
        dx = transform[0, 2]
        # translation y
        dy = transform[1, 2]
        # rotation
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0

    return [dx, dy, da]


def transform_frame(frame, transform, border_size, border_type):
    if border_type not in ['black', 'reflect', 'replicate']:
        raise ValueError('Invalid border type')

    transform = build_transformation_matrix(transform)
    bordered_frame_image, border_mode = border_frame(frame, border_size, border_type)

    h, w = bordered_frame_image.shape[:2]
    transformed_frame_image = cv2.warpAffine(bordered_frame_image, transform, (w, h), borderMode=border_mode)

    transformed_frame = Frame(transformed_frame_image, color_format='BGRA')

    return transformed_frame


def post_process_transformed_frame(transformed_frame, border_options, layer_options):
    cropped_frame = crop_frame(transformed_frame, border_options)

    if layer_options['layer_func'] is not None:
        cropped_frame = layer_utils.apply_layer_func(cropped_frame,
                                                     layer_options['prev_frame'],
                                                     layer_options['layer_func'])

        layer_options['prev_frame'] = cropped_frame

    return cropped_frame, layer_options





def bfill_rolling_mean(arr, n=30):
    """Helper to perform trajectory smoothing

    :param arr: Numpy array of frame trajectory to be smoothed
    :param n: window size for rolling mean
    :return: smoothed input arr

    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> bfill_rolling_mean(arr, n=2)
    array([[2.5, 3.5, 4.5],
           [2.5, 3.5, 4.5]])
    """
    if arr.shape[0] < n:
        raise ValueError('arr.shape[0] cannot be less than n')
    if n == 1:
        return arr

    pre_buffer = np.zeros(3).reshape(1, 3)
    post_buffer = np.zeros(3 * n).reshape(n, 3)
    arr_cumsum = np.cumsum(np.vstack((pre_buffer, arr, post_buffer)), axis=0)
    buffer_roll_mean = (arr_cumsum[n:, :] - arr_cumsum[:-n, :]) / float(n)
    trunc_roll_mean = buffer_roll_mean[:-n, ]

    bfill_size = arr.shape[0] - trunc_roll_mean.shape[0]
    bfill = np.tile(trunc_roll_mean[0, :], (bfill_size, 1))

    return np.vstack((bfill, trunc_roll_mean))



class PopDeque(deque):
    def deque_full(self):
        """Проверить, заполнена ли очередь"""
        return len(self) == self.maxlen

    def pop_append(self, x):
        """deque.append helper to return popped element if deque is at ``maxlen``

        :param x: element to append
        :return: result of ``deque.popleft()`` if deque is full; else ``None``

        >>> x = PopDeque([0], maxlen=2)
        >>> x.pop_append(1)

        >>> x.pop_append(2)
        0
        """
        popped_element = None
        if self.deque_full():
            popped_element = self.popleft()

        self.append(x)

        return popped_element

    def increment_append(self, increment=1, pop_append=True):
        """Append deque[-1] + ``increment`` to end of deque


        Если двухсторонняя очередь пуста, то добавляется 0

        :param increment: size of increment in deque
        :param pop_append: return popped element if append forces pop?
        :return: popped_element if pop_append is True; else None
        """
        if len(self) == 0:
            popped_element = self.pop_append(0)
        else:
            popped_element = self.pop_append(self[-1] + increment)

        if not pop_append:
            return None

        return popped_element


class Frame:
    """Утилита для упрощения преобразования цветовых форматов.

    :param image: OpenCV image as numpy array.
    :param color_format: Name of input color format or None.
         If str, the input must use the format that is used in OpenCV's cvtColor code parameter.
         For example, if an image is bgr then input 'BGR' as seen in the cvtColor codes:
        [cv2.COLOR_BGR2GRAY, COLOR_Luv2BGR].
        If None, the color format will be assumed from shape of the image.
        The only possible outcomes of this assumption are: ['GRAY', 'BGR', 'BGRA'].

    :ivar image: input image with possible color format conversions applied
    :ivar color_format: str containing the current color format of image attribute.
    """
    def __init__(self, image, color_format=None):
        self.image = image

        if color_format is None:
            self.color_format = self._guess_color_format()
        else:
            self.color_format = color_format

    def _guess_color_format(self):
        if len(self.image.shape) == 2:
            return 'GRAY'

        elif self.image.shape[2] == 3:
            return 'BGR'

        elif self.image.shape[2] == 4:
            return 'BGRA'

        else:
            raise ValueError(f'Unexpected frame image shape: {self.image.shape}')

    @staticmethod
    def _lookup_color_conversion(from_format, to_format):
        return getattr(cv2, f'COLOR_{from_format}2{to_format}')

    def cvt_color(self, to_format):
        if not self.color_format == to_format:
            color_conversion = self._lookup_color_conversion(from_format=self.color_format,
                                                             to_format=to_format)

            return cv2.cvtColor(self.image, color_conversion)
        else:
            return self.image

    @property
    def gray_image(self):
        return self.cvt_color('GRAY')

    @property
    def bgr_image(self):
        return self.cvt_color('BGR')

    @property
    def bgra_image(self):
        return self.cvt_color('BGRA')





class FrameQueue:
    def __init__(self, max_len=None, max_frames=None):
        self.max_len = max_len
        self.max_frames = max_frames
        self._max_frames = None

        self.frames = PopDeque(maxlen=max_len)
        self.inds = PopDeque(maxlen=max_len)
        self.i = None

        self.source = None
        self.source_frame_count = None
        self.source_fps = 30

        self.grabbed_frame = False

    def reset_queue(self, max_len=None, max_frames=None):
        self.max_len = max_len if max_len is not None else self.max_len
        self.max_frames = max_frames if max_frames is not None else self.max_frames

        has_max_frames = self.max_frames is not None and not np.isinf(self.max_frames)
        if has_max_frames:
            self._max_frames = self.max_frames

        self.frames = PopDeque(maxlen=max_len)
        self.inds = PopDeque(maxlen=max_len)
        self.i = None

    def set_frame_source(self, source):
        if isinstance(source, cv2.VideoCapture):
            self.source = source
            self.source_frame_count = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
            self.source_fps = int(source.get(cv2.CAP_PROP_FPS))

            has_max_frames = self.max_frames is not None and not np.isinf(self.max_frames)
            if self.source_frame_count > 0 and not has_max_frames:
                self._max_frames = self.source_frame_count
            elif has_max_frames and self.source_frame_count < self.max_frames:
                self._max_frames = self.source_frame_count
        else:
            raise TypeError('Not yet support for non cv2.VideoCapture frame source.')

    def read_frame(self, pop_ind=True, array=None):
        if isinstance(self.source, cv2.VideoCapture):
            self.grabbed_frame, frame = self.source.read()
        else:
            frame = array

        return self._append_frame(frame, pop_ind)

    def _append_frame(self, frame, pop_ind=True):
        popped_frame = None
        if frame is not None:
            popped_frame = self.frames.pop_append(Frame(frame))
            self.i = self.inds.increment_append()

        if pop_ind and self.i is None:
            self.i = self.inds.popleft()

        if (pop_ind
                and self.i is not None
                and self.max_frames is not None):
            break_flag = self.i >= self.max_frames
        else:
            break_flag = None

        return self.i, popped_frame, break_flag

    def populate_queue(self, smoothing_window):
        n = min([smoothing_window, self.max_frames])

        for i in range(n):
            _, _, _ = self.read_frame(pop_ind=False)
            if not self.grabbed_frame:
                break

    def frames_to_process(self):
        return len(self.frames) > 0 or self.grabbed_frame







class VidStab:
    """Класс для стабилизации видео

    В данном классе реализовано только стабилизация фрейма (на вход подаем вреймы) на выходе получаем стабилизированные фреймы)

    Процесс вычисляет оптический поток (cv2.calcOpticalFlowPyrLK) от кадра к кадру, используя
    ключевые точки, созданные детектором ключевых точек. Оптический поток будет
    использоваться для генерации преобразований кадров в кадры (cv2.estimateAffinePartial2D).
    Будут применены преобразования (cv2.warpAffine) для стабилизации видео.

    This class is based on the `work presented by Nghia Ho <http://nghiaho.com/?p=2093>`_

    :param kp_method: String of the type of keypoint detector to use. Available options are:
                        ``["GFTT", "BRISK", "DENSE", "FAST", "HARRIS", "MSER", "ORB", "STAR"]``.
                        ``["SIFT", "SURF"]`` are additional non-free options available depending
                        on your build of OpenCV.  The non-free detectors are not tested with this package.
    :param processing_max_dim: Working with large frames can harm performance (especially in live video).
                                   Setting this parameter can restrict frame size while processing.
                                   The outputted frames will remain the original size.

                                   For example:

                                   * If an input frame shape is `(200, 400, 3)` and `processing_max_dim` is
                                     100.  The frame will be resized to `(50, 100, 3)` before processing.

                                   * If an input frame shape is `(400, 200, 3)` and `processing_max_dim` is
                                     100.  The frame will be resized to `(100, 50, 3)` before processing.

                                   * If an input frame shape is `(50, 50, 3)` and `processing_max_dim` is
                                     100.  The frame be unchanged for processing.

    :param args: Positional arguments for keypoint detector.
    :param kwargs: Keyword arguments for keypoint detector.

    :ivar kp_method: a string naming the keypoint detector being used
    :ivar processing_max_dim: max image dimension while processing transforms
    :ivar kp_detector: the keypoint detector object being used
    :ivar trajectory: a 2d showing the trajectory of the input video
    :ivar smoothed_trajectory: a 2d numpy array showing the smoothed trajectory of the input video
    :ivar transforms: a 2d numpy array storing the transformations used from frame to frame
    """

    def __init__(self, kp_method='GFTT', processing_max_dim=float('inf'), *args, **kwargs):
        """Инициализиуем основной класс

        :param kp_method: В настоящий момент мы поддерживаем только 1 тип детектора особых точек GFTT. При необходимости можно лнгео добавть
                        ``["GFTT", "BRISK", "DENSE", "FAST", "HARRIS", "MSER", "ORB", "STAR"]``.
                        ``["SIFT", "SURF"]`` либо не свободные
                        но для этого нужно будет собирать версию cv2 именно с этими детекторами.
        :param processing_max_dim: Данный параметр должен влиять на производительность при работе с кадрами большого размера путем ограничивания размера кадра внутри класса.
                                    см пример ниже.
                                   For example:
                                     * If an input frame shape is `(200, 400, 3)` and `processing_max_dim` is
                                   100.  The frame will be resized to `(50, 100, 3)` before processing.
                                     * If an input frame shape is `(400, 200, 3)` and `processing_max_dim` is
                                   100.  The frame will be resized to `(100, 50, 3)` before processing.
                                     * If an input frame shape is `(50, 50, 3)` and `processing_max_dim` is
                                   100.  The frame be unchanged for processing.

        :param args: Позиционные аргументы для детектора ключевых точек (не поддероживается)
        :param kwargs: Параметры детектора ключевых точек (не поддердивается)
        """

        self.kp_method = kp_method
        # use original defaults in http://nghiaho.com/?p=2093 if GFTT with no additional (kw)args
        if kp_method == 'GFTT' and args == () and kwargs == {}:
            
            # self.kp_detector = kp_factory.FeatureDetector_create('GFTT',
            #                                                      maxCorners=200,
            #                                                      qualityLevel=0.01,
            #                                                      minDistance=30.0,
            #                                                      blockSize=3)
            self.kp_detector = GFTT(maxCorners=200,qualityLevel=0.01,minDistance=30.0,blockSize=3)
        #else:
        #    self.kp_detector = kp_factory.FeatureDetector_create(kp_method, *args, **kwargs)!!!!!!!!!!!!!!!!!!!!!!!

        self.processing_max_dim = processing_max_dim
        self._processing_resize_kwargs = {}

        self._smoothing_window = 30
        self._raw_transforms = []
        self._trajectory = []
        self.trajectory = self.smoothed_trajectory = self.transforms = None

        self.frame_queue = FrameQueue()
        self.prev_kps = self.prev_gray = None

        self.writer = None

        self.layer_options = {
            'layer_func': None,
            'prev_frame': None
        }

        self.border_options = {}
        self.auto_border_flag = False
        self.extreme_frame_corners = {'min_x': 0, 'min_y': 0, 'max_x': 0, 'max_y': 0}
        self.frame_corners = None

        self._default_stabilize_frame_output = None

    def _resize_frame(self, frame):
        if self._processing_resize_kwargs == {}:
            if self.processing_max_dim:
                shape = frame.shape
                max_dim_size = max(shape)

                if max_dim_size <= self.processing_max_dim:
                    self.processing_max_dim = max_dim_size
                    self._processing_resize_kwargs = None
                else:
                    max_dim_ind = shape.index(max_dim_size)
                    max_dim_name = ['height', 'width'][max_dim_ind]
                    self._processing_resize_kwargs = {max_dim_name: self.processing_max_dim}

        if self._processing_resize_kwargs is None:
            return frame

        resized = resize(frame, **self._processing_resize_kwargs)
        return resized

    def _update_prev_frame(self, current_frame_gray):
        self.prev_gray = current_frame_gray[:]
        self.prev_kps = self.kp_detector.detect(self.prev_gray)
        # noinspection PyArgumentList
        self.prev_kps = np.array([kp.pt for kp in self.prev_kps], dtype='float32').reshape(-1, 1, 2)

    def _update_trajectory(self, transform):
        if not self._trajectory:
            self._trajectory.append(transform[:])
        else:
            # gen cumsum for new row and append
            self._trajectory.append([self._trajectory[-1][j] + x for j, x in enumerate(transform)])

    #базовый класс который выполняет обработку
    def _gen_next_raw_transform(self):
        current_frame = self.frame_queue.frames[-1]
        current_frame_gray = current_frame.gray_image
        current_frame_gray = self._resize_frame(current_frame_gray)

        # Расчитаем оптический поток движения
        optical_flow = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                current_frame_gray,
                                                self.prev_kps, None)

        matched_keypoints = match_keypoints(optical_flow, self.prev_kps)
        transform_i = estimate_partial_transform(matched_keypoints)

        #
        # обновить информацию о предыдущем кадре для следующей итерации
        self._update_prev_frame(current_frame_gray)
        self._raw_transforms.append(transform_i[:])
        self._update_trajectory(transform_i)


    def _init_is_complete(self, gen_all):
        if gen_all:
            return False

        max_ind = min([self.frame_queue.max_frames,
                       self.frame_queue.max_len])

        if self.frame_queue.inds[-1] >= max_ind - 1:
            return True

        return False

    def _process_first_frame(self, array=None):
        # read first frame
        _, _, _ = self.frame_queue.read_frame(array=array, pop_ind=False)

        if array is None and len(self.frame_queue.frames) == 0:
            raise ValueError('First frame is None. Check if input file/stream is correct.')

        # convert to gray scale
        prev_frame = self.frame_queue.frames[-1]
        prev_frame_gray = prev_frame.gray_image
        prev_frame_gray = self._resize_frame(prev_frame_gray)

        # detect keypoints
        prev_kps = self.kp_detector.detect(prev_frame_gray)
        # noinspection PyArgumentList
        self.prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32').reshape(-1, 1, 2)

        self.prev_gray = prev_frame_gray[:]





    def _set_border_options(self, border_size, border_type):
        functional_border_size, functional_neg_border_size = functional_border_sizes(border_size)

        self.border_options = {
            'border_type': border_type,
            'border_size': functional_border_size,
            'neg_border_size': functional_neg_border_size,
            'extreme_frame_corners': self.extreme_frame_corners,
            'auto_border_flag': self.auto_border_flag
        }



    def _gen_transforms(self):
        self.trajectory = np.array(self._trajectory)
        self.smoothed_trajectory = bfill_rolling_mean(self.trajectory, n=self._smoothing_window)
        self.transforms = np.array(self._raw_transforms) + (self.smoothed_trajectory - self.trajectory)

        # Dump superfluous frames
        # noinspection PyProtectedMember
        n = self.frame_queue._max_frames
        if n:
            self.trajectory = self.trajectory[:n - 1, :]
            self.smoothed_trajectory = self.smoothed_trajectory[:n - 1, :]
            self.transforms = self.transforms[:n - 1, :]






    def _apply_next_transform(self, i, frame_i, use_stored_transforms=False):
        if not use_stored_transforms:
            self._gen_transforms()

        if i is None:
            i = self.frame_queue.inds.popleft()

        if frame_i is None:
            frame_i = self.frame_queue.frames.popleft()

        try:
            transform_i = self.transforms[i, :]
        except IndexError:
            return None

        transformed = transform_frame(frame_i,
                                                    transform_i,
                                                    self.border_options['border_size'],
                                                    self.border_options['border_type'])

        transformed, self.layer_options = post_process_transformed_frame(transformed,
                                                                                       self.border_options,
                                                                                       self.layer_options)

        transformed = transformed.cvt_color(frame_i.color_format)

        return transformed




    def stabilize_frame(self, input_frame, smoothing_window=30,
                        border_type='black', border_size=0, layer_func=None,
                        use_stored_transforms=False):
        """Стабилизировать один кадр из видео. Основная функция.
        Выполняет стабилизацию кадра за 1 раз. smoothing_window это задержка. Если задержка больше чем номер кадра будет возвращен черный кадр.

        :param input_frame: An OpenCV image (as numpy array) or None
        :param smoothing_window: window size to use when smoothing trajectory
        :param border_type: How to handle negative space created by stabilization translations/rotations.
                            Options: ``['black', 'reflect', 'replicate']``
        :param border_size: Size of border in output.
                            Positive values will pad video equally on all sides,
                            negative values will crop video equally on all sides,
                            ``'auto'`` will attempt to minimally pad to avoid cutting off portions of transformed frames
        :param layer_func: Function to layer frames in output.
                           The function should accept 2 parameters: foreground & background.
                           The current frame of video will be passed as foreground,
                           the previous frame will be passed as the background
                           (after the first frame of output the background will be the output of
                           layer_func on the last iteration)
        :param use_stored_transforms: should stored transforms from last stabilization be used instead of
                                      recalculating them?
        :return: 1 of 3 outputs will be returned:

            * Case 1 - Stabilization process is still warming up
                + **An all black frame of same shape as input_frame is returned.**
                + A minimum of ``smoothing_window`` frames need to be processed to perform stabilization.
                + This behavior was based on ``cv2.bgsegm.createBackgroundSubtractorMOG()``.
            * Case 2 - Stabilization process is warmed up and ``input_frame is not None``
                + **A stabilized frame is returned**
                + This will not be the stabilized version of ``input_frame``.
                  Stabilization is on an ``smoothing_window`` frame delay
            * Case 3 - Stabilization process is finished
                + **None**

        >>> from vidstab.VidStab import VidStab
        >>> stabilizer = VidStab()
        >>> vidcap = cv2.VideoCapture('input_video.mov')
        >>> while True:
        >>>     grabbed_frame, frame = vidcap.read()
        >>>     # Pass frame to stabilizer even if frame is None
        >>>     # stabilized_frame will be an all black frame until iteration 30
        >>>     stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
        >>>                                                   smoothing_window=30)
        >>>     if stabilized_frame is None:
        >>>         # There are no more frames available to stabilize
        >>>         break
        """
        self._set_border_options(border_size, border_type)
        self.layer_options['layer_func'] = layer_func
        self._smoothing_window = smoothing_window

        # Создаем первый кадр и возвращаем его (по факту мы добавляем в очередь наш кадр и возвращаем черный кадр)
        if self.frame_queue.max_len is None:
            self.frame_queue.reset_queue(max_len=smoothing_window + 1, max_frames=float('inf'))

            self._process_first_frame(array=input_frame)

            blank_frame = Frame(np.zeros_like(input_frame))
            blank_frame = crop_frame(blank_frame, self.border_options)

            self._default_stabilize_frame_output = blank_frame.image

            return self._default_stabilize_frame_output
        #если мы получили none в место кадра вы возвращаем none
        if len(self.frame_queue.frames) == 0:
            return None
        #мы добавляем наш кадр в очередеь
        frame_i = None
        if input_frame is not None:
            _, frame_i, _ = self.frame_queue.read_frame(array=input_frame, pop_ind=False)
            if not use_stored_transforms:
                self._gen_next_raw_transform()
        #проверяем некое условие и в случае его выполнения возвращаем черный квадрат
        if not self._init_is_complete(gen_all=False):
            return self._default_stabilize_frame_output

        stabilized_frame = self._apply_next_transform(self.frame_queue.i,
                                                      frame_i,
                                                      use_stored_transforms=use_stored_transforms)

        return stabilized_frame






if __name__ == "__main__":


         stabilizer = VidStab(processing_max_dim = 250)
         cap = cv2.VideoCapture(0)
         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
         cap.set(cv2.CAP_PROP_FPS, 30)
         ##############
         new_patch_video = 'video_out_43.avi'
         w = int(cap.get(3))
         h = int(cap.get(4))
         print(w, h)
         frame_fps = int(cap.get(5))
         vw = cv2.VideoWriter(new_patch_video, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), frame_fps,
                              (w * 2, h))
         counter = 0
         while (True):
             timer = cv2.getTickCount()
             # Capture frame-by-frame
             ret, frame = cap.read()
             #frame = cv2.flip(frame, -1)
             # Our operations on the frame come here
             # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) layer_func=layer_overlay
             stabilized_frame = stabilizer.stabilize_frame(input_frame=frame,
                                                           smoothing_window=1, border_type='replicate')
             # if np.max(stabilized_frame)==0:stabilized_frame=frame

             if stabilized_frame is None:
                 counter += 1
                 print(counter)
                 if counter > 10:
                     break
                 continue

             if frame is None:
                 counter += 1
                 print(counter)
                 if counter > 10:
                     break
                 continue
             vis = np.concatenate((frame, stabilized_frame), axis=1)

             fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
             if fps > 60:
                 myColor = (20, 230, 20)
             elif fps > 20:
                 myColor = (230, 20, 20)
             else:
                 myColor = (20, 20, 230)

             if int(fps) < 5:
                 stabilizer = None
                 stabilizer = VidStab(processing_max_dim=250)
                 print('RESTART_STAB')

             cv2.putText(vis, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2);
             cv2.putText(vis, "NOT STABILIZED", (int(w * 2 / 5), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 230), 2);
             cv2.putText(vis, "STABILIZED", (int(w * 2 / 1.5), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 20, 20), 2);

             vw.write(vis)
             # vis = cv2.resize(
             #    vis, (int(vis.shape[1] * 0.5),
             #            int(vis.shape[0] * 0.5)))
             # Display the resulting frame
             # print(vis.shape)

             cv2.imshow('frame', vis)

             if cv2.waitKey(1) & 0xFF == ord('q'):
                 break

         # When everything done, release the capture
         cap.release()
         cv2.destroyAllWindows()

