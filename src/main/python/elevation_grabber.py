from dataclasses import dataclass
import numpy as np
from typing import Iterator

#throws an error if value is not a power of two
def expect_power_of_2(val: float):
    from helper import is_int
    from math import log2

    if not is_int(log2(val)):
        raise ValueError(f"{val} must be a power of 2")


# This probably should have not been frozen due to performance penalties
# But making it not frozen breaks pre-existing files, unfortunately
@dataclass(slots=True, frozen=True)
class Coord:
    latitude: float
    longitude: float

    def of_scale(self, incr: float)->bool:
        return self.latitude % incr == 0 and self.longitude % incr == 0

    # The closer to the poles, the lower this ratio is
    # the lower the ratio, the more horizontally expanded the image is
    @property
    def ratio(self):
        from haversine import haversine

        dist_at_lat = haversine((self.latitude, 0), (self.latitude, 1))
        dist_at_equator = haversine((0, 0), (0, 1))

        return dist_at_lat / dist_at_equator

    def __add__(self, other: tuple[float,float])->"Coord":
        return Coord(
            self.latitude + other[0],
            self.longitude + other[1]
        )

#Represents a bounding box defined by two corners
@dataclass(slots=True)
class Box:
    min_coord: Coord
    max_coord: Coord

    def __init__(self, corner1: Coord, corner2: Coord):
        self.min_coord = Coord(
            latitude=min(corner1.latitude, corner2.latitude),
            longitude=min(corner1.longitude, corner2.longitude),
        )

        self.max_coord = Coord(
            latitude=max(corner1.latitude, corner2.latitude),
            longitude=max(corner1.longitude, corner2.longitude),
        )

    # Returns a new bounding box containing both this bounding box, and the new specified coordinate
    # Returns just this bounding box if the specified coordinate was within this bounding box
    def expand(self, c: Coord):
        return Box(
            Coord(
                min(self.min_coord.latitude, c.latitude),
                min(self.min_coord.longitude, c.longitude)
            ),
            Coord(
                max(self.max_coord.latitude, c.latitude),
                max(self.max_coord.longitude, c.longitude)
            )
        )

    # Returns whether a specified coordinate is within this bounding box
    def __contains__(self, c: Coord):
        return (
            (self.min_coord.latitude <= c.latitude <= self.max_coord.latitude) and
            (self.min_coord.longitude <= c.longitude <= self.max_coord.longitude)
        )

    # Returns the shape of a 2d numpy array representing this bounding box
    def arr_shape(self, scale: float)->tuple[int,int]:
        return (
            int((self.max_coord.latitude - self.min_coord.latitude + scale) / scale),
            int((self.max_coord.longitude - self.min_coord.longitude + scale) / scale)
        )

    # Converts a coordinate to a corresponding index to a 2d numpy array representing this bounding box
    def coord_to_index(self, c: Coord, scale: float)->tuple[int,int]:
        return (
            int((c.latitude - self.min_coord.latitude) / scale),
            int((c.longitude - self.min_coord.longitude) / scale),
        )


    # Yields all the coordinates of a specified scale within this bounding box
    def iterate(self, scale: float):
        expect_power_of_2(scale)

        if not self.min_coord.of_scale(scale) or not self.max_coord.of_scale(scale):
            raise Exception()

        for lat in np.arange(self.min_coord.latitude, self.max_coord.latitude + scale, scale):
            for lon in np.arange(self.min_coord.longitude, self.max_coord.longitude + scale, scale):
                yield Coord(lat, lon)


    # Returns a new bounding box with a specified height directly north of this bounding box
    def north(self, height: float):
        return Box(
            self.max_coord,
            Coord(
                self.max_coord.latitude + height,
                self.min_coord.longitude,
            )
        )

    # Returns a new bounding box with a specified height directly south of this bounding box
    def south(self, height: float):
        return Box(
            self.min_coord,
            Coord(
                self.min_coord.latitude - height,
                self.max_coord.longitude,
            )
        )

    # Returns a new bounding box with a specified width directly west of this bounding box
    def west(self, width: float):
        return Box(
            self.min_coord,
            Coord(
                self.max_coord.latitude,
                self.min_coord.longitude - width,
            )
        )

    # Returns a new bounding box with a specified width directly east of this bounding box
    def east(self, width: float):
        return Box(
            self.max_coord,
            Coord(
                self.min_coord.latitude,
                self.max_coord.longitude + width,
            )
        )


# Represents downloaded coordinate-to-elevation dataset
# All coordinates are a multiple of a specified scale
# This specified scale must be a power of 2, ie: 1/2, 1/8, 1/512
@dataclass(slots=True)
class Elevation:
    scale: float
    min_coord: Coord
    _data: np.ndarray

    base_url = "https://api.opentopodata.org/v1/ned10m?"

    def __init__(self, scale: float, min_coord: Coord, data: np.ndarray):
        self.scale = scale
        self.min_coord = min_coord

        self._data = data

    # Takes in a coordinate and returns the downloaded elevation
    # Returns None if the elevation was not downloaded at that position, or if the elevation is at or below sea-level
    def __getitem__(self, c: Coord) -> float|None:
        if not c.of_scale(self.scale):
            return None

        i = self.box.coord_to_index(c, self.scale)

        try:
            v = self._data[i]
            if np.isnan(v) or v <= 0:
                return None
            return v
        except IndexError:
            return None

    # Returns whether a coordinate was already downloaded
    def already_downloaded(self, c: Coord)->bool:
        if not c.of_scale(self.scale):
            return False

        i = self.box.coord_to_index(c, self.scale)

        try:
            v = self._data[i]

            return not np.isnan(v)
        except IndexError:
            return False

    #Returns the entire bounding box of all downloaded elevation
    @property
    def box(self):
        return Box(
            self.min_coord,
            self.min_coord + (self._data.shape[0]*self.scale, self._data.shape[1]*self.scale),
        )

    # Converts the index to the image representing this dataset into a coordinate
    def index_to_coord(self, i: tuple[int,int]) -> Coord:
        return self.box.index_to_coord(i, self.scale)

    # Downloads all the elevation from a list of coordinates and returns a new dataset with those new elevations
    def download(self, new_coords: Iterator[Coord] | list[Coord], new_scale: float) -> "Elevation":
        import requests
        import time
        from helper import split
        from threading import Thread, Lock
        from tqdm import tqdm
        from math import isnan

        expect_power_of_2(new_scale)

        # Directly downloads and returns a list of elevations corresponding to a list of coordinates
        # Returns None if there is an error retrieving the data
        def download_from_http(coords: list[Coord], session) -> None | Iterator[float]:
            if len(coords) == 0:
                return (_ for _ in ())
            if len(coords) > 100:
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
                    self.base_url,
                    params={"locations": a, "interpolation": "cubic"}
                )
            except requests.exceptions.RequestException as e:
                print(e)
                return None

            if response.status_code != 200:
                print(response.reason)
                return None

            return (r["elevation"] for r in response.json()["results_"])

        # Filters out coordinates that are already downloaded
        def filter_new_coords():
            nonlocal new_coords

            new_coords = [c for c in new_coords if not self.already_downloaded(c)]

        # Creates a new bounding box with all the new coordinates
        def make_new_box():
            new_box = self.box

            for c in new_coords:
                new_box = new_box.expand(c)

            return new_box

        # Fills the new dataset with pre-existing elevation data
        def fill_with_old_data():
            for i, v in np.ndenumerate(self._data):
                c = self.box.index_to_coord(i, self.scale)

                new_i = new_box.coord_to_index(c, new_scale)

                new_data[new_i] = v

            return new_data

        # Fills the new dataset with new elevation data
        def fill_with_new_data():
            import itertools
            from threading import Lock, Thread, active_count

            session = requests.session()

            bulk = split(new_coords, max_len=100)

            bar = tqdm(total=len(bulk), smoothing=0)

            for i in itertools.count():
                time.sleep(1)

                def try_run(coords: list[Coord]):
                    downloaded = download_from_http(coords, session)

                    if downloaded is None:
                        # If fails, then put it back to the end of the list to be re-downloaded later
                        bulk.append(coords)
                    else:
                        for c, e in zip(coords, downloaded):
                            new_data[new_box.coord_to_index(c, new_scale)] = e

                        bar.update()

                with Lock():
                    if i >= len(bulk):
                        break

                    Thread(target=lambda: try_run(bulk[i])).start()

                if i % 2048 == 0:
                    print(active_count())

        filter_new_coords()

        new_box = make_new_box()

        new_data = np.full(shape=new_box.arr_shape(new_scale), fill_value=np.nan, dtype=float)
        fill_with_old_data()
        fill_with_new_data()

        return Elevation(
            min_coord=new_box.min_coord,
            scale=new_scale,
            data=new_data,
        )

    #Returns an image representing this elevation dataset
    #TODO come up with an understandable explanation for the border parameter
    def to_img(self, border: float | None = None)->None:
        from helper import to_png

        img = self._data.copy()

        if border is not None:
            for (px, py), v in np.ndenumerate(img):
                lat = self.min_coord.latitude + px*self.scale
                lon = self.min_coord.longitude + py*self.scale

                if lat % border == 0 or lon % border == 0:
                    img[px, py] = np.nan

        return img

    # Converts the elevation data to multiple images to be fed to the AI model
    # Applies data augmentation, such as reflection and rotations
    # TODO, comment inside this function more
    def to_ai_dataset(self, img_len: int, steps: int) -> list[np.ndarray]:
        from math import sqrt, ceil
        from scipy.ndimage import rotate

        shape = self.box.arr_shape(self.scale)

        def run(img_len: int)->Iterator[np.ndarray]:
            px, py = 0, 0
            while px + img_len < shape[0]:
                ratio = self.index_to_coord((px, py)).ratio

                # img_len is the unexpanded width
                expanded_width = int(img_len / ratio)
                expanded_steps = int(steps / ratio)

                py = 0
                while py + expanded_width < shape[1]:
                    img = self._data[px:px + img_len, py:py + expanded_width]

                    if not np.isnan(img).any():
                        import cv2

                        # unexpands the images horizontally
                        img = cv2.resize(img, (img_len, img_len), interpolation=cv2.INTER_AREA)

                        yield img

                    py += steps

                px += expanded_steps


        def rot_45_then_crop(img: np.ndarray)->np.ndarray:
            from helper import crop_to_len

            img = rotate(img, 45, reshape=False)
            img = crop_to_len(img, img_len)

            return img

        def all_transformation(img: np.ndarray)->Iterator[np.ndarray]:
            for r in range(4):
                yield np.rot90(img, r)
            for r in range(4):
                yield np.fliplr(np.rot90(img, r))
            # yield img

        for img in run(img_len):
            for img_t in all_transformation(img):
                yield img_t

        for img in run(ceil(img_len * sqrt(2)) + 1):
            for img_t in all_transformation(img):
                yield rot_45_then_crop(img_t)