from helper import NxN, iter_square, load
from generate_world import make_elevation_map
import numpy as np

if __name__ == '__main__':
    total_len = 64
    rng = np.random.default_rng(0)

    base = make_elevation_map(
        models=[load("in/desert_16")],
        biome_map=[NxN.ones(total_len)],
        total_len=total_len,
        paste_len=32,
        rng=rng,
    )

    base.to_png("result")

    i = 0

    for (xi, yi) in iter_square(total_len, steps=32):
        base.copy(xi, yi, 32).to_png(f"out/desert_16/{i}")
        i += 1