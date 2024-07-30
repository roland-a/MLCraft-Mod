import struct
from dataclasses import dataclass
import numpy as np
from typing import Iterator, Callable
from helper import ByteConsumer, ByteProducer
from functools import cached_property


# The closer to the poles, the lower this ratio is
# the lower the ratio, the more horizontally expanded the image is
def get_ratio(latitude: float)->float:
    from haversine import haversine

    dist_at_lat = haversine((latitude, 0), (latitude, 1))
    dist_at_equator = haversine((0, 0), (0, 1))

    return dist_at_lat / dist_at_equator

# Represents a real life coordinate
@dataclass(slots=True, frozen=True)
class Coord:
    latitude: float
    longitude: float

    def __getitem__(self, item):
        if item==0: return self.latitude
        if item==1: return self.longitude

        raise Exception(f"{item}")

    def __iter__(self):
        yield self.longitude
        yield self.longitude


# Performs a one-to-one mapping from a subset of N floating points to an integer from 0 to N
# Does this efficiently when the subset of values and when the distance between the members are a power of 2
@dataclass(slots=True, frozen=True)
class Indexer:
    min_value: float
    max_value: float
    jump: float

    # Constructs an Indexer that can map all the floats within an iterator to an index
    @staticmethod
    def from_list(lst: Iterator[float]):
        jump = 1

        min_value = float("inf")
        max_value = float("-inf")

        for x in lst:
            if x < min_value:
                min_value = x
            if x > max_value:
                max_value = x

            while x % result != 0:
                jump /= 2

        return Indexer(
            min_value=min_value,
            max_value=max_value,
            jump=jump,
        )

    # Constructs an Indexer from a sequence of bytes produced by a ByteProducer
    @staticmethod
    def from_bytes(producer: ByteProducer):
        from helper import float32_from_bytes

        return Indexer(
            min_value=float32_from_bytes(producer),
            max_value=float32_from_bytes(producer),
            jump=float32_from_bytes(producer),
        )

    #Gets the index associated with a floating point value
    def value_to_index(self, x: float)->int:
        return int((x-self.min_value)/self.jump)

    # Gets the floating point value associated with an index
    def index_to_value(self, x: int)->float:
        return x*self.jump + self.min_value

    # Returns the maximum index
    def max_index(self):
        return self.value_to_index(self.max_value)

    # Feeds a sequence of byte that represents this instance into a ByteConsumer
    def to_bytes(self, consumer: ByteConsumer):
        from helper import float32_to_bytes

        float32_to_bytes(self.min_value, consumer)
        float32_to_bytes(self.max_value, consumer)
        float32_to_bytes(self.jump, consumer)


    # Returns True if this Indexer can map argument x into an index
    # Returns False otherwise
    def __contains__(self, x: float):
        return self.max_value >= x >= self.min_value and x % self.jump == 0


# A data-structure represents a map that associates a point to a float
# Uses two Indexer objects to map each point to an index that corresponds to a numpy array entry
@dataclass(slots=True, frozen=True)
class PointToFloat:
    x_indexer: Indexer
    y_indexer: Indexer
    _data: np.ndarray

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    # Constructs a point-to-float map from a point-to-float dict
    @staticmethod
    def from_dict(map: dict[tuple[int,int], float]):
        x_indexer = Indexer.from_list(c[0] for c in map.keys())
        y_indexer = Indexer.from_list(c[1] for c in map.keys())

        shape = (
            x_indexer.max_index()+1,
            y_indexer.max_index()+1
        )

        # Entries without an associated value are NaN
        data = np.full(shape=shape, dtype=np.float32, fill_value=np.nan)

        for c, v in map.items():
            i = (
                x_indexer.value_to_index(c[0]),
                y_indexer.value_to_index(c[1]),
            )
            data[i] = v

        return PointToFloat(
            x_indexer=x_indexer,
            y_indexer=y_indexer,
            _data=data
        )

    @staticmethod
    def from_bytes(producer: ByteProducer):
        from helper import int32_from_bytes, float32_from_bytes, np_from_bytes

        return PointToFloat(
            x_indexer=Indexer.from_bytes(producer),
            y_indexer=Indexer.from_bytes(producer),
            _data=np_from_bytes(producer)
        )

    # Returns the value that is associated with the specified point
    # Returns None if the specified point is not associated with a value
    def __getitem__(self, point: tuple[float, float])->float|None:
        if point[0] not in self.x_indexer:
            return None
        if point[1] not in self.y_indexer:
            return None

        i = (
            self.x_indexer.value_to_index(point[0]),
            self.y_indexer.value_to_index(point[1]),
        )

        if i[0] > self._data.shape[0]:
            return None
        if i[1] > self._data.shape[1]:
            return None

        v = self._data[i]
        if np.isnan(v):
            return None
        return v

    # Returns True if a point has an associated value
    # Returns False otherwise
    def __contains__(self, point: tuple[float, float])->float:
        return self[point] is not None

    # Iterates through all points and their associated values
    def __iter__(self)->Iterator[tuple[tuple[float, float], float]]:
        for i, v in np.ndenumerate(self._data):
            if np.isnan(v):
                continue

            c = (
                self.x_indexer.index_to_float(i[0]),
                self.x_indexer.index_to_float(i[1]),
            )

            yield c, v

    def to_bytes(self, consumer: ByteConsumer):
        from helper import int32_to_bytes, float32_to_bytes, np_to_bytes

        self.x_indexer.to_bytes(consumer)
        self.y_indexer.to_bytes(consumer)

        np_to_bytes(self._data, consumer)


# Splits the list into batches, with each batch's size at most max_len
def split(lst: list, max_len: int)->list[list]:
    # class Return:
    #     def __getitem__(self, i):
    #         return Chunk(start=i*max_len)
    #
    #     def __len__(self):
    #         from math import ceil
    #
    #         return int(ceil(len(lst) / max_len))
    #
    #     def __iter__(self):
    #         for i in range(len(self)):
    #             yield self[i]
    #
    # class Chunk:
    #     def __init__(self, start):
    #         self.start = start
    #
    #     def __getitem__(self, i):
    #         return lst[self.start + (i % max_len)]
    #
    #     def __len__(self):
    #         return int(min(len(lst) - self.start, max_len))
    #
    #     def __iter__(self):
    #         for i in range(len(self)):
    #             yield self[i]
    #
    # return Return()

    assert max_len > 0

    bulks = [[]]

    for v in lst:
        if len(bulks[-1]) >= max_len:
            bulks.append([])

        bulks[-1].append(v)

    return bulks


# Takes a list of coordinates and returns a dict that contains the inputted coordinates as keys, and the elevation at that coordinate as the associated value
def download(coords: list[Coord], base_url: str, max_attempts = 100)->dict[Coord, float]:
    import requests
    import time
    from threading import Thread, Lock
    from tqdm import tqdm
    from math import isnan
    import itertools

    # Directly downloads and returns a list of elevations corresponding to a list of coordinates
    # Returns None if there is an error retrieving the data
    def download_from_http(coords: list[Coord], session) -> None | Iterator[float]:
        if len(coords) == 0:
            return (_ for _ in ())
        if len(coords) > max_attempts:
            raise Exception("Too many inputs")

        # Converts the list of dataset into a HTTP POST request parameter
        # TODO put this into a separate function
        a = ""
        first = True
        for c in coords:
            if not first:
                a += "|"

            a += str(c.latitude)
            a += ","
            a += str(c.longitude)
            first = False

        try:
            response = session.get(
                base_url,
                params={"locations": a, "interpolation": "cubic"}
            )
        except requests.exceptions.RequestException as e:
            print(e)
            return None

        if response.status_code != 200:
            print(response.reason)
            return None

        return (r["elevation"] for r in response.json()["results"])

    def run_parallel(index: int):
        nonlocal completed
        nonlocal bar

        downloaded = download_from_http(bulk[index], session)

        if downloaded is not None:
            for c, e in zip(bulk[index], downloaded):
                results[c] = e

            completed[i] = True
            bar.update()

    session = requests.session()

    results = {}

    bulk = split(coords, max_len=max_attempts)
    completed = [False for _ in bulk]

    bar = tqdm(total=len(completed), smoothing=0)

    try:
        i = 0
        while completed.count(False) > 0:
            i = i % len(completed)

            if not completed[i]:
                Thread(target=lambda: run_parallel(i)).start()
                time.sleep(1)

            i += 1
    except Exception as e:
        print(e)

    return results
