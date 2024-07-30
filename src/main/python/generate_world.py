import dataclasses
import numpy as np
from denoising_diffusion_pytorch import GaussianDiffusion
from typing import Callable
from matplotlib import pyplot as plt
from helper import to_png

# Slices a numpy array into a smaller array, while wrapping around if the slice exceeds the boundary
def slice_wrap_around(arr: np.ndarray, offset: tuple, out_shape: int|tuple)->np.ndarray:
    if isinstance(out_shape, int):
        out_shape = (out_shape, ) * len(arr.shape)

    result = np.zeros(shape=out_shape, dtype=arr.dtype)

    for i, _ in np.ndenumerate(result):
        i_off = tuple((a+b)%s for a, b, s in zip(i, offset, arr.shape))

        i = tuple(a % s for a, s in zip(i, result.shape))

        result[i] = arr[i_off]

    return result


# Offsets the right array, wraps the right array around the left array, then performs actions between the two array where they overlap
def offset_then_apply_wraparound(left, right, right_offset: tuple, fn)->np.ndarray:
    arr = left.copy()

    for i, _ in np.ndenumerate(right):
        i_off = tuple((i+o) % s for i, o, s in zip(i, right_offset, left.shape))

        arr[i_off] = fn(left[i_off], right[i])

    return arr


# Interpolates a 2d numpy array while respecting wraparound boundaries
def wrap_around_interpolate(arr: np.ndarray, factor: int)->np.ndarray:
    if arr.dtype == int:
        arr = np.repeat(arr, factor, axis=0)
        arr = np.repeat(arr, factor, axis=1)

        return arr

    from cv2 import resize, INTER_CUBIC, INTER_LANCZOS4

    pad_len = 8 * factor, 8 * factor

    # Adds wrap-around padding to effectively allow wraparound interpolation to the non-padded areas
    arr = np.pad(arr, pad_len, mode="wrap")

    arr = resize(arr, (arr.shape[1] * factor, arr.shape[0] * factor), interpolation=INTER_LANCZOS4)

    # Removes the padding
    # TODO, use crop_to_len to remove padding
    arr = arr[pad_len[0] * factor:-pad_len[0] * factor, pad_len[1] * factor:-pad_len[1] * factor]

    return arr


# Returns how many 1000x1000 sections there are in a 2d numpy array
def n_1000x1000(arr: np.ndarray) -> float:
    assert len(arr.shape) == 2

    return np.prod(arr.shape) / (1000**2)


# Converts a 2d numpy array into a pure binary file at a specified path
# The array's width and height must be equal, so that the image length can be determined by the square root of the number of entries
def to_binary(arr: np.ndarray, path: str):
    from helper import normalize_min_max
    import struct

    assert len(arr.shape) == 2
    assert arr.shape[0] == arr.shape[1]

    arr = normalize_min_max(
        arr,
        old_min_max=(np.nanmin(arr), np.nanmax(arr)),
        new_min_max=(0, 255)
    )

    arr = arr.astype(dtype=np.uint8)

    result = bytearray()

    for (i, v) in np.ndenumerate(arr):
        result += struct.pack("B", v)

    with open(path, "wb") as f:
        f.write(result)


# Returns a simple biome map with N distinct biomes
# Does not support groups like oceans/land or snowy/medium/hot yet
def make_biome_map(
    n_biomes: int,
    map_len: int,
    n_layers: int,
    points_per_layer: Callable[[int], int],
    rng: np.random.Generator
)->np.ndarray:
    from itertools import product

    point = tuple[float, float]

    def hash_point(p: point) -> int:
        import xxhash
        import struct

        x, y = p

        return xxhash.xxh32_intdigest(struct.pack('<f', x) + struct.pack('<f', y), seed=0)

    def get_dist_sq(p1: point, p2: point)->float:
        x1, y1 = p1
        x2, y2 = p2

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        if dx > .5:
            dx = 1 - dx
        if dy > .5:
            dy = 1 - dy

        return dx**2 + dy**2

    def get_closest(p: point, points: list[point]) -> point:
        return min(
            points,
            key=lambda ps: get_dist_sq(p, ps)
        )

    def traverse_points(p: point, layers: list[list[point]]) -> point:
        for layer in reversed(layers):
            p = get_closest(p, layer)

        return p

    def make_layers(rng) -> list[list[point]]:
        return [
            make_layer(points_per_layer(i), rng) for i in range(n_layers)
        ]

    def make_layer(n_points: int, rng) -> list[point]:
        return [
            (rng.random(), rng.random()) for _ in range(n_points)
        ]

    layers = make_layers(rng)

    base = np.zeros(shape=(map_len, map_len), dtype=int)

    for (x, y) in product(range(map_len), repeat=2):
        p = (x / map_len, y / map_len)

        p = traverse_points(p, layers)

        b = hash_point(p) % n_biomes

        base[x, y] = b

    return base


# Throws an assertion error if there is an issue with denoising the image
def assert_denoisable(img: np.ndarray, denoise_map: np.ndarray, padding_len: int, denoisers) -> bool:
    denoise_len = denoisers[0].image_size - padding_len * 2

    assert img.shape == denoise_map.shape
    assert not np.isnan(img).any()

    assert len(denoise_map.shape) == 2
    assert all(x % denoise_len == 0 for x in img.shape)
    assert np.min(denoise_map) >= 0
    assert np.max(denoise_map) < len(denoisers)
    assert denoise_map.dtype == int

    assert all(x.image_size == denoisers[0].image_size for x in denoisers)
    assert all(x.num_timesteps == denoisers[0].num_timesteps for x in denoisers)


#Performs one step of denoising on a section of the array, then returns the results
def denoise(
    img: np.ndarray,
    denoise_map: np.ndarray,
    denoisers: list[GaussianDiffusion],
    padding_len: int,
    grid_offset: int,
    step: int
):
    def run_step(model, img, t):
        import torch
        from helper import crop_to_len

        type = img.dtype

        model.eval()

        with torch.inference_mode():
            img = torch.from_numpy(img).float()

            # Turns the [N,N] array to [1,1,N,N]
            # First 1 is for batch size, second 1 is for channel size
            img = img.unsqueeze(0).unsqueeze(0)

            img, _ = model.p_sample(
                img,
                model.num_timesteps-1-t
            )

            # Turns the [1,1,N,N] array back to [N,N]
            img = img.squeeze(0).squeeze(0)

            img = img.numpy()
            img = img.astype(type)

            # Crops the final output to the denoise length
            img = crop_to_len(img, denoise_len)

            return img

    denoise_len = denoisers[0].image_size - padding_len*2

    result = np.zeros_like(img)

    assert_denoisable(result, denoise_map, padding_len, denoisers)

    for x, y in product(
        range(0, img.shape[0], denoise_len),
        range(0, img.shape[1], denoise_len)
    ):
        x += grid_offset
        y += grid_offset

        for i, b in enumerate(denoisers):
            # grabs a section of the denoise map
            b_section = slice_wrap_around(
                denoise_map,
                offset=(x + padding_len, y + padding_len),
                out_shape=denoise_len
            )

            # prevents unneeded computation if the biome is nowhere on the section
            if np.all(b_section != i):
                continue

            # grabs a section of the current image, plus padding
            section = slice_wrap_around(
                img,
                offset=(x - padding_len, y - padding_len),
                out_shape=denoise_len + padding_len * 2
            )

            # slightly denoises that section
            # the padding is automatically cropped out
            section = run_step(b, section, step)

            # zeros out the location where the biome is not at
            section *= (b_section == i)

            # adds that section back
            result = offset_then_apply_wraparound(
                left=result,
                right=section,
                right_offset=(x, y),
                fn=lambda l, r: l + r
            )
    return result


# Returns a unique image according to a denoiser map with a list of denoisers
def make_init_img(
    rng: np.random.Generator,
    denoise_map: np.ndarray,
    denoisers: list[GaussianDiffusion],
    padding_len: int,
):
    img = rng.uniform(size=denoise_map.shape).astype(np.float32)

    assert_denoisable(img, denoise_map, padding_len, denoisers)

    denoise_steps = denoisers[0].num_timesteps
    denoise_len = denoisers[0].image_size - padding_len*2
    
    for t in tqdm(range(denoise_steps)):
        grid_offset = (denoise_steps // denoise_len) * t

        to_png(img, path=f"tmp/tmp/{t}")

        img = denoise(
            img=img,
            denoise_map=denoise_map, 
            denoisers=denoisers, 
            grid_offset=grid_offset, 
            padding_len=padding_len, 
            step=t
        )
        
    return img


# Creates a new image similar to an initial image according to a denoiser map with a list of denoisers
# Essentially noises the initial image to a certain degree, and denoises it according to the denoiser map
def img_to_img(
    init_img: np.ndarray,
    denoise_map: np.ndarray,
    denoisers: list[GaussianDiffusion],
    padding_len: int,
    similarity: float
):
    denoise_steps = denoisers[0].num_timesteps
    denoise_len = denoisers[0].image_size - padding_len*2

    step = round(similarity * denoise_steps)

    #renoises the image
    init_img = denoisers[0].q_sample(
        x_start=torch.from_numpy(init_img).unsqueeze(0).unsqueeze(0),
        t=torch.tensor(denoisers[0].num_timesteps - 1 - step).unsqueeze(0)
    ).squeeze(0).squeeze(0).numpy()

    #denoises the image
    for t in tqdm(range(step, denoise_steps)):
        grid_offset = ((denoise_steps-step) // denoise_len) * t

        init_img = denoise(
            img=init_img,
            denoise_map=denoise_map, 
            denoisers=denoisers, 
            grid_offset=grid_offset,
            padding_len=padding_len,
            step=t
        )

    return init_img
