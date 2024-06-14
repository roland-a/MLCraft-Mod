import pickle
import numpy as np
from typing import Callable


# Unpickles a python object from a file
def load(path: str):
    import joblib
    
    with open(f"{path}.joblib", 'rb') as file:
        return joblib.load(file)


# Pickles a python object into a file
# TODO, remove this function and instead use dedicated serializing, as pickling makes refactoring much harder
def save(obj: any, path: str):
    import joblib
    
    write_with_backup(path, "joblib", lambda p: joblib.dump(obj, open(p, "wb")))
    
    
# Converts a 2d numpy array to a greyscale png with 16-bit depth
def to_png(arr: np.ndarray, path: str, min_max=None):
    import cv2

    if min_max is not None:
        min, max = min_max
    else:
        min, max = np.nanmin(arr), np.nanmax(arr)


    if np.nanmin(arr) < min or np.nanmax(arr) > max:
        raise Exception(f"{min, max} must be wider than {np.nanmin(arr), np.nanmax(arr)}")

    arr = arr.astype(dtype=float)
    arr = np.nan_to_num(arr, nan=min)
    arr -= min
    arr /= (max-min)

    arr = (arr * (2**16-1)).astype(dtype=np.uint16)

    write_with_backup(
        path, "png", lambda p: cv2.imwrite(p, arr)
    )
    
    
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


# Returns whether a specified float is exactly an integer
def is_int(v):
    return v == int(v)

#splits the list into batches, with a batch size at most max_len
def split(lst: list, max_len: int)->list[list]:
    bulks = [[]]

    for v in lst:
        if len(bulks[-1]) >= max_len:
            bulks.append([])

        bulks[-1].append(v)

    return bulks

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


# Slices a 2d numpy array into a smaller 2d array, while wrapping around if the slice exceeds the boundary
def slice_wrap_around(arr: np.ndarray, offset: tuple, slice_len: int)->np.ndarray:
    out = np.empty(shape=(slice_len, slice_len), dtype=arr.dtype)

    for i, _ in np.ndenumerate(out):
        i = tuple(i%s for i, s in zip(i, arr.shape))
        i_off = tuple((a+b)%s for a, b, s in zip(i, offset, arr.shape))

        out[i] = arr[i_off]

    return out

# TODO, refactor this function so its more understandable
def apply_wrap_around(left, right, right_offset: tuple, fn)->np.ndarray:
    arr = left.copy()

    for i, _ in np.ndenumerate(right):
        i_off = tuple((i+o) % s for i, o, s in zip(i, right_offset, left.shape))

        arr[i_off] = fn(left[i_off], right[i])

    return arr


# Interpolates a 2d numpy array while respecting wraparound boundaries
def wrap_around_interpolate(arr: np.ndarray, inter_len: int)->np.ndarray:
    import cv2

    pad_len = 8 * inter_len

    # Adds wrap-around padding to effectively allow wraparound interpolation to the non-padded areas
    arr = np.pad(arr, (pad_len, pad_len), mode='wrap')

    arr = cv2.resize(arr, (arr.shape[0] * inter_len, arr.shape[1] * inter_len), interpolation=cv2.INTER_LANCZOS4)

    # Removes the padding
    # TODO, use crop_to_len to remove padding
    arr = arr[pad_len * inter_len:-pad_len * inter_len, pad_len * inter_len:-pad_len * inter_len]

    return arr