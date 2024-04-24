import itertools
import pickle

import cv2
import numpy as np
from PIL import Image


def load(path: str):
    return pickle.load(open(path, 'rb'))


def save(obj: any, path: str):
    pickle.dump(obj, open(path, 'wb'))


def iter_square(*args, steps=1):
    if len(args) == 1:
        mx = args[0]
        return itertools.product(range(0, mx, steps), range(0, mx, steps))
    elif len(args) == 2:
        (mn, mx) = args
        return itertools.product(range(mn, mx, steps), range(mn, mx, steps))
    else:
        raise Exception("Invalid number of arguments")


def into_pieces(arr: [], n: int):
    result = [[]]

    for a in arr:
        if len(result[-1]) > n:
            result.append([])

        result[-1].append(a)

    return result


class NxN:
    def __init__(self, arr: np.ndarray):
        if len(arr.shape) != 2: raise Exception(f"Must be 2d, instead shape is {arr.shape}")
        if arr.shape[0] != arr.shape[1]: raise Exception(f"Must be Square, instead shape is {arr.shape}")

        self.arr = arr

    @staticmethod
    def zero(length: int) -> "NxN":
        return NxN(np.zeros(shape=(length, length), dtype=float))

    @staticmethod
    def ones(length: int) -> "NxN":
        return NxN(np.ones(shape=(length, length), dtype=float))

    @staticmethod
    def random(length: int, rng) -> "NxN":
        return NxN(rng.normal(size=(length, length)))

    @property
    def len(self) -> int:
        return len(self.arr)

    @property
    def min(self) -> int:
        return np.min(self.arr)

    @property
    def max(self) -> int:
        return np.max(self.arr)

    def is_zero(self) -> bool:
        return not np.any(self.arr)

    def __eq__(self, other):
        raise Exception()

    def __mul__(self, other) -> "NxN":
        if type(other).__name__ == "NxN":
            return self.binary_op(other, lambda x, y: x * y)

        return NxN(self.arr * other)

    def __add__(self, other: "NxN") -> "NxN":
        return self.binary_op(other, lambda a, b: a + b)

    def binary_op(self, other: "NxN", op) -> "NxN":
        if self.len % other.len != 0 and other.len % self.len != 0:
            raise Exception(f"arrays must be multiples of eachother: {self.len}, {other.len}")

        length = max(self.len, other.len)

        out = np.zeros(shape=(length, length), dtype=float)

        for (x, y) in iter_square(length):
            out[x][y] = op(self[x, y], other[x, y])

        return NxN(out)

    def __getitem__(self, item: (int, int)) -> float:
        return self.arr[item[0] % self.len][item[1] % self.len]

    def copy(self, start_x: int, start_y: int, length: int) -> "NxN":
        out = np.zeros(shape=(length, length), dtype=float)

        for (x, y) in iter_square(length):
            out[x][y] = self[(start_x + x) % self.len, (start_y + y) % self.len]

        return NxN(out)

    def paste(self, r: "NxN", start_x: int, start_y: int) -> "NxN":
        arr = self.arr.copy()

        for x in range(r.len):
            for y in range(r.len):
                arr[(start_x + x) % self.len][(start_y + y) % self.len] = r[x, y]

        return NxN(arr)

    def lerp(self, other: "NxN", p: float) -> "NxN":
        return (self * (1 - p)) + (other * p)

    def interpolate(self, length: int) -> "NxN":
        pad_len = 8 * length

        arr = np.pad(self.arr, (pad_len, pad_len), mode='wrap')

        arr = cv2.resize(arr, (len(arr) * length, len(arr) * length), interpolation=cv2.INTER_LANCZOS4)

        arr = arr[pad_len * length:-pad_len * length, pad_len * length:-pad_len * length]

        return NxN(arr)

    def border(self, length: int) -> "NxN":
        return NxN(
            np.pad(self.arr[length:-length, length:-length], pad_width=(length, length), constant_values=0)
        )

    def to_mc_format(self, name: str) -> "NxN":
        import struct

        mn = np.min(self.arr)
        mx = np.max(self.arr)

        file = open(name, "wb")
        for (x, y) in iter_square(self.len):
            v = self[x, y]

            v -= mn
            v /= (mx - mn)
            v *= 65535
            v = int(v)

            file.write(struct.pack(">H", v))
        file.close()

        return self

    def to_png(self, name: str, min_max: (int, int) = None):
        if min_max is None:
            mn = np.min(self.arr)
            mx = np.max(self.arr)
        else:
            mn, mx = min_max

        image = Image.new("L", size=(self.len, self.len))
        for x in range(self.len):
            for y in range(self.len):
                v = self.arr[x][y]

                v -= mn
                v /= (mx - mn)
                v *= 255
                v = int(v)

                image.putpixel((x, y), v)
        image.save(f"{name}.png")

        return self

    def test_wrap(self) -> "NxN":
        return self.copy(self.len * 3 // 2, self.len * 3 // 2, self.len)
