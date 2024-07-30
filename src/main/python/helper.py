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

    
# Moves old data about to be rewritten into a backup folder before writing a new file
def write_with_backup(path, ext, on_path):
    import time
    import os

    parent = "/".join(path.split("/")[:-1])
    file = f"{path}.{ext}"

    # Make the parent directory if it does not exist
    if parent != "":
        os.makedirs(
            parent,
            exist_ok=True
        )

    # If there is already a file under this path, then move it into the backup
    if os.path.isfile(file):
        backup_parent = "tmp/backup/" + parent

        os.makedirs(
            backup_parent,
            exist_ok=True
        )

        # Add a unique suffix to the moved file, so it can't overwrite any other file
        suffix = round(time.time())
        backup_file = f"backup/{path}-{suffix}.{ext}"

        os.rename(
            file,
            backup_file
        )

    on_path(file)


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

    imwrite(path + ".png", arr)
