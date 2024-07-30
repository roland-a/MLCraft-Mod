import pickle
import numpy as np
from typing import Callable


ByteConsumer = Callable[[bytes], None]
ByteProducer = Callable[[int], bytes]

_struct_i32 = Struct("<i")
_struct_f32 = Struct("<f")


# Returns a ByteProducer that returns sequences of bytes from a file of a specified path
def produce_bytes(path: str):
    class Reader:
        def __call__(self, *args, **kwargs):
            assert len(args) == 1
            assert isinstance(args[0], int)
            assert len(kwargs) == 0

            return self.file.read(args[0])

        def __del__(self):
            self.file.close()

    reader = Reader()
    reader.file = open(path, 'rb')
    return reader


def int32_from_bytes(producer: ByteProducer)->int:
    return _struct_i32.unpack(producer(4))[0]


def float32_from_bytes(producer: ByteProducer)->float:
    return _struct_f32.unpack(producer(4))[0]


def np_from_bytes(producer: ByteProducer)->np.ndarray:
    shape_len = int32_from_bytes(producer)
    shape = []

    for _ in range(shape_len):
        shape.append(
            int32_from_bytes(producer)
        )

    count = int(np.prod(shape))
    size = count * 4

    arr = np.frombuffer(producer(size), dtype=np.float32, count=count)

    arr = arr.reshape(shape)

    return arr


# Returns a ByteConsumer that accepts sequences of bytes and places it in a file of a specified path
def consume_bytes(path: str):
    class Writer:
        def __call__(self, *args, **kwargs):
            assert len(args) == 1
            assert isinstance(args[0], bytes | bytearray)
            assert len(kwargs) == 0

            self.file.write(args[0])

        def __del__(self):
            self.file.close()


    writer = Writer()
    writer.file = open(path, 'wb')

    return writer


def int32_to_bytes(f: float, consumer: ByteConsumer)->None:
    consumer(
        _struct_i32.pack(f)
    )


def float32_to_bytes(f: float, consumer: ByteConsumer)->None:
    consumer(
        _struct_f32.pack(f)
    )


def np_to_bytes(arr: np.ndarray, consumer: ByteConsumer)->None:
    assert arr.dtype == np.float32

    int32_to_bytes(len(arr.shape), consumer)

    for s in arr.shape:
        int32_to_bytes(s, consumer)

    consumer(arr.tobytes())


#Takes in a numpy array of an old min-max, then return a version that's rescaled to a new min-max
def normalize_min_max(arr: np.ndarray, old_min_max: tuple[int, int], new_min_max: tuple[int, int])->np.ndarray:
    arr = arr.copy()

    arr = (arr - old_min_max[0]) / (old_min_max[1] - old_min_max[0])

    arr = arr * (new_min_max[1] - new_min_max[0]) + new_min_max[0]

    return arr


# Crops a 2d numpy image to a specified length
def crop_to_len(img: np.ndarray, final_len: int|tuple[int,int]) -> np.ndarray:
    if isinstance(final_len, int):
        final_len = (final_len, final_len)

    if img.shape == final_len:
        return img

    left_pad_x = (img.shape[0] - final_len[0])//2
    left_pad_y = (img.shape[1] - final_len[1])//2

    if left_pad_x<0 or left_pad_y<0:
        raise Exception()

    right_pad_x = (img.shape[0] - final_len[0])-left_pad_x
    right_pad_y = (img.shape[1] - final_len[1])-left_pad_y

    if left_pad_x != 0 and right_pad_x != 0:
        img = img[left_pad_x:-right_pad_x, :]

    if left_pad_y != 0 and right_pad_y != 0:
        img = img[:, left_pad_y:-right_pad_y]

    return img


# Returns a random section of a numpy array
# This section will also be of a random rotation and will be randomly mirrored
def random_section(
    arr: np.ndarray,
    section_len: int,
    rng: np.random.Generator=np.random.default_rng()
)->np.ndarray:
    from math import sqrt, pi, ceil, sin, cos, degrees
    from scipy.ndimage import rotate
    from helper import crop_to_len

    assert min(arr.shape) >= section_len * sqrt(2) > 0

    angle = rng.uniform(0, 2*pi)

    pre_rot_len = ceil((section_len * (abs(cos(angle)) + abs(sin(angle)))))+1

    px_start = rng.integers(
        low=0,
        high=arr.shape[0]-pre_rot_len
    )

    py_start = rng.integers(
        low=0,
        high=arr.shape[1]-pre_rot_len
    )

    should_flip = rng.integers(low=0, high=1)

    img = arr[
        px_start:px_start + pre_rot_len,
        py_start:py_start + pre_rot_len,
    ]

    if should_flip:
        img = np.fliplr(img)

    img = rotate(
        input=img,
        angle=degrees(angle),
        order=1
    )

    return crop_to_len(img, section_len)

def prepare_before_overwrite(path: str):
    import time
    import os

    path = path.replace("/", os.sep)

    parent = os.sep.join(path.split(os.sep)[:-1])

    if parent != "":
        os.makedirs(
            parent,
            exist_ok=True
        )

    os.makedirs(
        "backup" + os.sep + parent,
        exist_ok=True
    )

    if os.path.isfile(path):
        return

    # TODO create backup files
    # # Add a unique suffix to the moved file, so it can't overwrite any other file
    # suffix = str(round(time.time()))
    # backup_path = "backup" + os.sep + path + suffix
    #
    # try:
    #     os.rename(
    #         path,
    #         backup_path
    #     )
    # except FileExistsError as e:
    #     print(e)
    #     pass


# Returns a numpy array from a png file with a specified min-max
def from_png(path: str, min_max:tuple[int,int]=(0,1))->np.ndarray:
    import cv2
    import numpy as np

    img = cv2.imread(path, -1)

    if img.dtype == np.uint8:
        old_min_max = (0, 2**8-1)
    elif img.dtype == np.uint16:
        old_min_max = (0, 2**16-1)
    else:
        raise Exception()

    img = normalize_min_max(
        img,
        old_min_max=old_min_max,
        new_min_max=min_max
    )
    img = img.astype(dtype=np.float32)

    return img

# Converts a 2d numpy array to a greyscale png
def to_png(arr: np.ndarray, path: str, min_max: tuple[int,int]=None, out=np.uint16):
    from cv2 import imwrite
    import os

    if min_max is not None:
        min_max = min_max
    else:
        min_max = np.nanmin(arr), np.nanmax(arr)

    if np.nanmin(arr) < min_max[0] or np.nanmax(arr) > min_max[1]:
        raise Exception(f"{min_max[0], min_max[1]} must be wider than {np.nanmin(arr), np.nanmax(arr)}")

    if min_max[0] == min_max[1]:
        move_before_overwrite(
            path + ".png"
        )

        imwrite(path + ".png", np.zeros_like(arr, dtype=out))

    arr = arr.copy()
    arr = arr.astype(dtype=np.float64)
    arr = np.nan_to_num(arr, nan=min_max[0])
    arr = normalize_min_max(
        arr,
        old_min_max=min_max,
        new_min_max=(0, np.iinfo(out).max)
    )
    arr = arr.astype(dtype=out)

    move_before_overwrite(path + ".png")

    imwrite(path + ".png", arr)
