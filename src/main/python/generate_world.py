import dataclasses
import time

import numpy as np
import torch
from denoising_diffusion_pytorch import GaussianDiffusion
from PIL import Image

from helper import NxN, iter_square, load, save, into_pieces

@dataclasses.dataclass
class BiomeInfo:
    model: GaussianDiffusion
    amp: float
    short_amp: float
    rbg: (int,int,int)

    @staticmethod
    def default_biomes() -> ["BiomeInfo"]:
        return [
            BiomeInfo(model=load("in/desert_16"), amp=1/8, short_amp=1/128, rbg=(255, 255, 0)),
            BiomeInfo(model=load("in/appalachians_16"), amp=1, short_amp=1/64, rbg=(128, 128, 128)),
        ]

def make_biome_map(
    n_biomes: int,
    total_len: int,
    n_layers: int,
    points_per_layer,
    rng
) -> [NxN]:
    import numpy as np
    from helper import iter_square, NxN

    def hash_point(p: (float, float)) -> int:
        import xxhash
        import struct

        x, y = p

        return xxhash.xxh32_intdigest(struct.pack('<f', x) + struct.pack('<f', y), seed=0)

    def get_closest(p: (float, float), points: [(float, float)]) -> (float, float):
        m = float("inf")
        mp = None

        x, y = p

        for p in points:
            for (xi, yi) in iter_square(-1, 1):
                xp, yp = p

                xp += xi * total_len
                yp += yi * total_len

                d = (x - xp) ** 2 + (y - yp) ** 2

                if d < m:
                    m = d
                    mp = p
        return mp

    def traverse_points(p: (float, float), layers: [[(float, float)]]) -> (float, float):
        for layer in reversed(layers):
            p = get_closest(p, layer)

        return p

    def make_layer(n_points: int, rng) -> [(float, float)]:
        layer = []

        for _ in range(n_points):
            x = rng.random() * total_len
            y = rng.random() * total_len

            layer.append((x, y))

        return layer

    def make_layers(rng) -> [[(float, float)]]:
        layers = []
        for i in range(n_layers):
            n_points = points_per_layer(i)

            layer = make_layer(n_points, rng)

            layers.append(layer)

        return layers

    def make_map(layers: [[(float, float)]]) -> [NxN]:
        from helper import iter_square

        base = np.zeros(shape=(n_biomes, total_len, total_len), dtype=int)

        for p in iter_square(total_len):
            x, y = p

            tp = traverse_points(p, layers)

            b = hash_point(tp) % n_biomes

            base[b][x][y] = 1

        return [NxN(arr) for arr in base]

    layers = make_layers(rng)

    return make_map(layers)


def biome_map_to_mc(id: int, biome_map: [NxN]):
    import struct

    file = open(f"out/biome{id}", mode="wb")

    n_biomes = len(biome_map)
    total_len = biome_map[0].len

    arr = np.zeros(shape=(total_len, total_len), dtype=float)
    for (x, y) in iter_square(total_len):
        out = None

        for i in range(n_biomes):
            if biome_map[i][x, y] == 1:
                out = i

        if out is None:
            raise Exception()

        arr[x][y] = out

    arr = NxN(arr)
    arr = arr.interpolate(16 // 4)

    for (x, y) in iter_square(total_len * (16 // 4)):
        out = round(arr[x, y])

        file.write(struct.pack(">B", out))


def biome_map_to_img(file_name: str, biome_map: [NxN], biomes: [BiomeInfo]):
    n_biomes = len(biome_map)
    total_len = biome_map[0].len

    img = Image.new("RGB", (total_len, total_len))

    for (x, y) in iter_square(total_len):
        r = 0
        g = 0
        b = 0

        for i in range(n_biomes):
            r += biome_map[i][x, y] * biomes[i].rbg[0]
            g += biome_map[i][x, y] * biomes[i].rbg[1]
            b += biome_map[i][x, y] * biomes[i].rbg[2]

        r = int(r)
        b = int(b)
        g = int(g)

        img.putpixel((x,y), (r,g,b))

    img.save(file_name)


def smooth_biome_map(base: [NxN], border_length: int) -> [NxN]:
    if border_length == 0:
        return base

    n_biomes = len(base)
    length = base[0].len

    ret = np.zeros(shape=(n_biomes, length, length))

    for x, y in iter_square(length):
        for k in range(n_biomes):
            sum = 0
            total = 0

            for (xk, yk) in iter_square(-border_length // 2, border_length // 2):
                if xk ** 2 + yk ** 2 > (border_length // 2) ** 2:
                    continue

                sum += base[k][x + xk, y + yk]
                total += 1

            ret[k][x][y] = sum / total

    return [NxN(arr) for arr in ret]


def run_step(self: NxN, model: GaussianDiffusion, step: int):
    arr = torch.from_numpy(
        np.array([[self.arr]], dtype=np.float32),
    )

    out, _ = model.p_sample(
        arr,
        model.num_timesteps - 1 - step,
        None
    )

    return NxN(out[0][0].numpy())


def make_elevation_map(
    models: [GaussianDiffusion],
    biome_map: [NxN],
    paste_len: int,
    total_len: int,
    rng,
):
    from tqdm import tqdm
    from helper import iter_square, NxN

    image_len = models[0].image_size
    time_steps = models[0].num_timesteps

    if (image_len - paste_len) % 2 != 0: raise Exception(f"{image_len}, {paste_len}")

    pad_len = (image_len - paste_len) // 2

    img = NxN.random(total_len, rng)

    for t in tqdm(range(0, time_steps)):
        out = NxN.zero(total_len)

        for (xi, yi) in iter_square(0, total_len, steps=paste_len):
            xi += t * (paste_len // time_steps)
            yi += t * (paste_len // time_steps)

            for i, model in enumerate(models):
                r = img.copy(xi - pad_len, yi - pad_len, image_len)

                b = biome_map[i].copy(xi, yi, paste_len)

                if b.is_zero():
                    continue

                r = run_step(r, model, t)

                r = r.copy(pad_len, pad_len, paste_len)

                r = r * b

                r = NxN.zero(total_len).paste(r, xi, yi)

                out += r

        img = out

    return img

# def make_base_blk(
#     models: [GaussianDiffusion],
#     biome_map: [NxN],
#     paste_len: int,
#     total_len: int,
#     rng,
#     pieces: int = 64
# ):
#     from tqdm import tqdm
#     from helper import iter_square, NxN
#
#     image_len = models[0].image_size
#     time_steps = models[0].num_timesteps
#
#     if (image_len - paste_len) % 2 != 0: raise Exception(f"{image_len}, {paste_len}")
#
#     pad_len = (image_len - paste_len) // 2
#
#     img = NxN.random(total_len, rng)
#
#     steps = (paste_len // time_steps)
#
#     for t in tqdm(range(0, time_steps), total=time_steps, position=0):
#         out = NxN.zero(total_len)
#
#         for i, model in enumerate(models):
#             blk_c = []
#             blk_b = []
#
#             blk_in = []
#             blk_out = []
#
#             for (xi, yi) in iter_square(total_len, steps=paste_len):
#                 xi += t * steps
#                 yi += t * steps
#
#                 b = biome_map[i].copy(xi, yi, paste_len)
#                 if b.is_zero():
#                     continue
#
#                 inp = img.copy(xi - pad_len, yi - pad_len, image_len)
#
#                 blk_c.append((xi, yi))
#                 blk_b.append(b)
#                 blk_in.append(inp)
#
#                 for b in tqdm(into_pieces(blk_in, pieces), position=1, disable=(len(blk_in) >= pieces)):
#                     blk_out += run_step_bulk(model, b, t)
#
#             for ((xi, yi), b, r) in zip(blk_c, blk_b, blk_out):
#                 r = r.copy(pad_len, pad_len, paste_len)
#
#                 r = r * b
#
#                 out += NxN.zero(total_len).paste(r, xi, yi)
#
#         img = out
#
#     return img
# def run_step_bulk(model: GaussianDiffusion, blk: [NxN], step: int) -> [NxN]:
#     model.eval()
#
#     with torch.no_grad():
#         tensor = torch.stack([torch.from_numpy(b.arr) for b in blk]).type(torch.FloatTensor)
#
#         tensor = tensor.reshape((len(tensor), 1, model.image_size, model.image_size))
#
#         tensor, _ = model.p_sample(
#             tensor,
#             model.num_timesteps - 1 - step,
#             None
#         )
#
#         tensor = tensor.reshape((len(tensor), model.image_size, model.image_size))
#
#         return [NxN(t.numpy()) for t in tensor]

def generate_world(
    id: int,
    total_len: int,
    rng,
    biomes: [BiomeInfo] = BiomeInfo.default_biomes(),
    elevation_paste_len: int = 32,
    elevation_interpolation: int = 32,
    noisy_map_total_len: int = 64,
    noisy_map_paste_len: int = 32,
    noisy_map_interpolation: int = 4,
    biome_n_layers: int = 4,
    biome_points_per_layer=lambda x: 16 * 2 ** x,
):
    start = time.time()

    print("Making biome map...")
    biome_map = make_biome_map(
        n_biomes=len(biomes),
        total_len=total_len,
        n_layers=biome_n_layers,
        points_per_layer=biome_points_per_layer,
        rng=rng,
    )

    biome_map_to_img(f"out/biome_map{id}.png", biome_map, biomes)
    biome_map_to_mc(id, biome_map)

    print("Smoothing biome map...")
    smoothed_biome_map = smooth_biome_map(base=biome_map, border_length=16)
    biome_map_to_img(f"out/biome_map_smoothed{id}.png", biome_map, biomes)

    print("Making elevation map...")
    base = make_elevation_map(
        models=[biome.model for biome in biomes],
        biome_map=biome_map,
        paste_len=elevation_paste_len,
        total_len=total_len,
        rng=rng,
    )
    base.to_png(f"out/init_elevation{id}")

    print("Making noisy maps...")
    noisy_maps = [
        make_elevation_map(
            models=[biomes[i].model],
            biome_map=[NxN.ones(noisy_map_total_len)],
            paste_len=noisy_map_paste_len,
            total_len=noisy_map_total_len,
            rng=rng,
        ) for i in range(len(biomes))
    ]
    for i, sb in enumerate(noisy_maps):
        sb.to_png(f"out/noisy_map{i}-{id}")

    print("Making amplitude map...")
    amp_map = NxN.zero(total_len)
    for (i, a) in enumerate([biome.amp for biome in biomes]):
        amp_map += smoothed_biome_map[i] * a
    amp_map.to_png(f"out/amp_map{id}")

    print("Interpolating elevation map...")
    elevation = base * amp_map
    elevation = elevation.interpolate(elevation_interpolation)

    print("Adding noisy maps to elevation map...")
    for (i, sb) in enumerate(noisy_maps):
        elevation += sb.interpolate(noisy_map_interpolation) * smoothed_biome_map[i] * biomes[i].short_amp

    elevation.to_png(f"out/elevation{id}").to_mc_format(f"out/elevation{id}")
    print("Done!")

    end = time.time()

    elapsed = end-start

    print(f"Time to generate: {elapsed}, Seconds per 1000x1000 blocks: {elapsed / (total_len*elevation_interpolation)**2/1000**2}")

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    generate_world(
        id=2,
        total_len=128,
        rng=rng
    )
