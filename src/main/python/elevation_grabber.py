import json
import time

import numpy
import requests
import numpy as np
from PIL import Image

from helper import NxN, iter_square


def get_altitudes(coords, session):
    result = []

    while True:
        base_url = "https://api.open-elevation.com/api/v1/lookup"

        a = ""

        first = True
        for c in coords:
            if not first:
                a += "|"

            a += str(c[0])
            a += ","
            a += str(c[1])
            first = False

        response = session.get(
            base_url,
            params={"locations": a}
        )

        try:
            for r in response.json()["results"]:
                result.append(r["elevation"])
            return result
        except:
            print(response.url)
            return None


def generate_map(lat_start: float, lng_start: float, incr: float, map_len:int):
    init_cooldown = .01
    init_bulk_length = 256

    result = numpy.zeros(shape=(map_len, map_len))

    all_xy = []
    for x in range(map_len):
        for y in range(map_len):
            all_xy.append((x, y))

    session = requests.session()

    i = 0
    while i < len(all_xy):
        attempt = 0
        while True:
            bulk_length = int(init_bulk_length * 2 ** (-attempt))
            if bulk_length < 2:
                bulk_length = 2

            bulk_xy = all_xy[i:min(i + bulk_length, len(all_xy))]
            bulk_coords = [(lat_start + x * incr, lng_start + y * incr) for (x, y) in bulk_xy]

            cool_down = init_cooldown * 2 ** attempt
            if cool_down > 30:
                cool_down = 30

            time.sleep(cool_down)

            bulk_altitude = get_altitudes(bulk_coords, session)

            if bulk_altitude is None:
                attempt += 1

                if attempt > 4:
                    session = requests.session()
                continue

            for j, r in enumerate(bulk_altitude):
                x, y = bulk_xy[j]

                result[x][y] = r

            print(f"{i / (map_len * map_len) * 100}% done")
            i += bulk_length
            break

    return NxN(result)

def run():
    incr = .002
    image_len = 64
    min_max_search = 10

    total_len = incr * image_len * min_max_search



    i = 1
    for lat in np.arange(47.3, 46.6, -total_len):
        for long in np.arange(-69.9, -66.7, total_len):
            map = generate_map(lat, long, incr, image_len*min_max_search)

            for (xi, yi) in iter_square(image_len*min_max_search, steps=image_len):
                img = map.copy(xi, yi, image_len)

                img.to_png(f"desert/{i}", min_max=(map.min, map.max))
                i += 1

    i = 1000
    for lat in np.arange(42.0, 39.8, -total_len):
        for long in np.arange(-79.8, -74.1, total_len):
            map = generate_map(lat, long, incr, image_len*min_max_search)

            for (xi, yi) in iter_square(image_len*min_max_search, steps=image_len):
                img = map.copy(xi, yi, image_len)

                img.to_png(f"desert/{i}", min_max=(map.min, map.max))
                i += 1

    i = 2000
    for lat in np.arange(37.6, 35.8, -total_len):
        for long in np.arange(-87.3, -81.0, total_len):
            map = generate_map(lat, long, incr, image_len*min_max_search)

            for (xi, yi) in iter_square(image_len*min_max_search, steps=image_len):
                img = map.copy(xi, yi, image_len)

                img.to_png(f"desert/{i}", min_max=(map.min, map.max))
                i += 1


if __name__ == "__main__":
    run()