#!/usr/bin/env python
"""Planets: a slowly rotating planet, lit from one side, against twinkling
stars. Cycles through Earth (clouds, city lights on the night side), the Moon
(in its real current phase), Mars, Jupiter (Great Red Spot) and Saturn (rings
passing in front of and behind it), fading between them.

Each pixel's spot on the sphere, its lighting and its anti-aliased edge are
worked out once per planet; a frame then just slides the planet's surface
texture past those spots, blending between texels so the rotation glides.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random
import time

import numpy as np

from matrix_common import REFRESH_HZ, run_scene

FRAME_SECONDS = 2 / REFRESH_HZ  # every 2nd refresh, ~37.5 fps

PLANET_SECONDS = 60.0
FADE_SECONDS = 1.5

# Surfaces are equirectangular textures: columns are longitude -180..180,
# rows latitude 90..-90
TEX_W, TEX_H = 128, 64
SUPERSAMPLE = 4  # samples per pixel per axis, for anti-aliased edges

STAR_COUNT = 40
STAR_TINTS = [(255, 255, 255), (190, 210, 255), (255, 230, 190)]

# Earth's land at 10-degree resolution: rows 90N..90S, columns 180W..180E
WORLD = [
    "............####....................",
    "......##########...#...############.",
    ".########.#.###.#.##################",
    "...##########....###############.#..",
    ".....########....################...",
    "......#####......################...",
    "......###.......###############.....",
    ".......###......#######..#.##.#.....",
    "..........###....######.....###.....",
    "..........#####....###......######..",
    "..........#####....####.......###...",
    "...........###.....####......#####..",
    "...........##......##........#####.#",
    "...........##...................#.#.",
    "...........#........................",
    "...........##.....##################",
    "..##########...####################.",
    "####################################",
]

# Synodic month and a known new moon (2000-01-06 18:14 UTC), for the phase
SYNODIC_DAYS = 29.530588853
NEW_MOON_EPOCH = 947182440


# --- Texture helpers ---------------------------------------------------------


def upsample(grid: np.ndarray, w: int = TEX_W, h: int = TEX_H) -> np.ndarray:
    """Smoothly interpolate a coarse (rows, cols) grid up to (h, w),
    wrapping around in longitude."""
    gh, gw = grid.shape
    x = np.arange(w) * gw / w
    y = np.linspace(0, gh - 1, h)
    x0 = np.floor(x).astype(int)
    y0 = np.minimum(np.floor(y).astype(int), gh - 2)
    fx = x - x0
    fy = y - y0
    fx = (fx * fx * (3 - 2 * fx))[None, :]
    fy = (fy * fy * (3 - 2 * fy))[:, None]
    x1 = (x0 + 1) % gw
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx
    bottom = grid[y0 + 1][:, x0] * (1 - fx) + grid[y0 + 1][:, x1] * fx
    return (top * (1 - fy) + bottom * fy).astype(np.float32)


def fbm(rng: np.random.Generator, base: int = 4, octaves: int = 4, stretch: float = 1.0) -> np.ndarray:
    """Fractal value noise in [0, 1] over the texture. stretch > 1 makes
    features longer east-west (for gas giant streaks)."""
    out = np.zeros((TEX_H, TEX_W), dtype=np.float32)
    total = 0.0
    for octave in range(octaves):
        cells = base * 2 ** octave
        amp = 0.5 ** octave
        out += upsample(rng.random((cells + 1, max(2, int(cells * 2 / stretch))))) * amp
        total += amp
    return out / total


def latitudes() -> np.ndarray:
    """Latitude in degrees of each texture row, as a column vector."""
    return np.linspace(90, -90, TEX_H, dtype=np.float32)[:, None]


def longitudes() -> np.ndarray:
    return np.linspace(-180, 180, TEX_W, endpoint=False, dtype=np.float32)[None, :]


def mix(a, b, t) -> np.ndarray:
    """Blend colors a -> b by t (arrays broadcast; t gets a color axis)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return a + (b - a) * np.asarray(t, dtype=np.float32)[..., None]


def bands(stops: list[tuple[float, tuple]], lat: np.ndarray) -> np.ndarray:
    """Colors by latitude, interpolated between (latitude, color) stops
    listed north to south."""
    lats = np.array([s[0] for s in stops], dtype=np.float32)[::-1]
    colors = np.array([s[1] for s in stops], dtype=np.float32)[::-1]
    lat = np.clip(lat, -90, 90)
    return np.stack([np.interp(lat, lats, colors[:, c]) for c in range(3)], axis=-1).astype(np.float32)


# --- Surfaces: (texture, extras) for each body --------------------------------


def earth_surface(rng):
    lat, lon = latitudes(), longitudes()
    world = np.array([[1.0 if ch == "#" else 0.0 for ch in row] for row in WORLD], dtype=np.float32)
    coast = upsample(world)
    shape = coast + (fbm(rng, base=6) - 0.5) * 0.45
    land = shape > 0.5

    detail = fbm(rng, base=8)
    green = mix((25, 105, 35), (80, 130, 45), detail)
    desert_band = np.clip(1 - abs(abs(lat) - 25) / 10, 0, 1) * (detail > 0.35)
    ground = mix(green, (200, 165, 105), desert_band)
    ground = mix(ground, (70, 90, 55), np.clip((abs(lat) - 52) / 10, 0, 1))
    ice = (lat > 72) | (lat < -62) | ((lat > 60) & (lon > -55) & (lon < -20))
    ground = np.where(ice[..., None], np.float32(230), ground)
    ocean = mix((0, 22, 80), (0, 60, 140), np.clip(shape * 2.2 - 0.2, 0, 1))
    texture = np.where(land[..., None], ground, ocean)

    # City lights: sparse warm dots on land, most at mid northern latitudes
    density = np.where((lat > 15) & (lat < 58), 0.16, 0.05)
    lights = (land & ~ice & (rng.random((TEX_H, TEX_W)) < density)).astype(np.float32)
    lights = lights[..., None] * np.array([255, 190, 100], dtype=np.float32)

    clouds = fbm(rng, base=5, octaves=5)
    clouds = np.clip((clouds - 0.5) * 3.4, 0, 0.9) * (1 - 0.6 * desert_band)
    return texture, {"lights": lights, "clouds": clouds}


def moon_surface(rng):
    lat, lon = latitudes(), longitudes()
    base = mix((150, 150, 145), (185, 185, 180), fbm(rng, base=10))
    # Dark maria, mostly on the side facing Earth (longitude near 0)
    near = np.clip(np.cos(np.radians(lon)) * 1.4, 0, 1)
    maria = np.clip((fbm(rng, base=3) - 0.48) * 6, 0, 1) * near * np.clip(1 - abs(lat) / 70, 0, 1)
    texture = mix(base, (85, 85, 92), maria * 0.85)
    # Craters: darker floors, bright rims
    y, x = np.mgrid[0:TEX_H, 0:TEX_W]
    for _ in range(70):
        cy, cx, r = rng.uniform(6, TEX_H - 6), rng.uniform(0, TEX_W), rng.uniform(0.8, 3.0)
        dx = np.minimum(abs(x - cx), TEX_W - abs(x - cx))
        d = np.hypot(dx, y - cy) / r
        texture *= np.where(d < 0.8, 0.82, np.where(d < 1.15, 1.18, 1.0))[..., None]
    # Tycho: a young bright crater in the southern highlands
    tycho = np.hypot((lon + 11) / 6, (lat + 43) / 4) < 1
    texture = np.where(tycho[..., None], np.float32(235), texture)
    return np.clip(texture, 0, 255), {}


def mars_surface(rng):
    lat = latitudes()
    albedo = fbm(rng, base=3, octaves=5)
    texture = mix((205, 110, 60), (175, 80, 45), np.clip(albedo * 1.5 - 0.3, 0, 1))
    texture = mix(texture, (100, 50, 35), np.clip((albedo - 0.55) * 4, 0, 1))  # dark regions
    caps = abs(lat) + (fbm(rng, base=8) - 0.5) * 10 > 74
    return np.where(caps[..., None], np.float32(240), texture), {}


def jupiter_surface(rng):
    lat, lon = latitudes(), longitudes()
    # Turbulence: bend the bands with noise stretched east-west
    wobble = (fbm(rng, base=6, octaves=4, stretch=4.0) - 0.5) * 7
    stops = [
        (90, (140, 130, 120)), (55, (170, 150, 125)), (38, (215, 200, 170)), (30, (175, 120, 80)),
        (22, (230, 215, 180)), (13, (180, 110, 70)), (6, (235, 220, 190)), (-6, (225, 205, 170)),
        (-12, (170, 100, 60)), (-20, (230, 210, 175)), (-28, (180, 130, 85)), (-40, (215, 200, 170)),
        (-58, (170, 150, 125)), (-90, (140, 130, 120)),
    ]
    texture = bands(stops, lat + wobble)
    # The Great Red Spot, with a pale collar
    d = np.hypot(lon / 13, (lat + 22) / 6)
    texture = mix(texture, (235, 205, 170), np.clip((1.35 - d) * 4, 0, 1))
    texture = mix(texture, (200, 85, 55), np.clip((1.0 - d) * 4, 0, 1))
    return texture, {}


def saturn_surface(rng):
    lat = latitudes()
    wobble = (fbm(rng, base=6, octaves=3, stretch=4.0) - 0.5) * 4
    stops = [
        (90, (150, 145, 120)), (60, (190, 175, 130)), (35, (215, 195, 145)), (20, (200, 175, 125)),
        (8, (235, 215, 160)), (-8, (230, 210, 155)), (-20, (205, 180, 130)), (-40, (215, 195, 145)),
        (-90, (160, 150, 120)),
    ]
    return bands(stops, lat + wobble), {}


# (inner, outer radius in planet radii, color, opacity), inside out
SATURN_RINGS = [
    (1.24, 1.52, (120, 110, 90), 0.35),   # C ring
    (1.52, 1.95, (230, 210, 165), 0.95),  # B ring
    (2.03, 2.27, (195, 180, 145), 0.75),  # A ring, past the Cassini division
]


# --- Rendering ---------------------------------------------------------------


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class Planet:
    """One body, precomputed for drawing at the middle of the panel."""

    def __init__(self, width: int, height: int, surface, *, radius: float, spin_seconds: float,
                 light, tilt: float = 0.0, roll: float = 0.0, oblate: float = 1.0,
                 ambient: float = 0.03, limb: float = 0.0, atmosphere=None, rings=None,
                 libration: float = 0.0, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        texture, extras = surface(rng)
        self.texture = texture.reshape(-1, 3).astype(np.float32)
        self.lights = extras.get("lights")
        if self.lights is not None:
            self.lights = self.lights.reshape(-1, 3)
        clouds = extras.get("clouds")
        self.clouds = None if clouds is None else clouds.reshape(-1).astype(np.float32)
        self.spin = TEX_W / spin_seconds if spin_seconds else 0.0  # texture columns per second
        self.libration = libration
        self.phase = random.uniform(0, TEX_W)  # start facing a random longitude

        # Every pixel that the planet, its rings or its glow can touch, with
        # SUPERSAMPLE x SUPERSAMPLE sample points each
        reach = radius * (SATURN_RINGS[-1][1] if rings else 1.2) + 1
        cx, cy = width / 2, height / 2
        xs = np.arange(max(0, int(cx - reach)), min(width, int(cx + reach) + 1))
        ys = np.arange(max(0, int(cy - reach)), min(height, int(cy + reach) + 1))
        py, px = (a.reshape(-1) for a in np.meshgrid(ys, xs, indexing="ij"))
        sub = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE
        oy, ox = (a.reshape(-1) for a in np.meshgrid(sub, sub, indexing="ij"))
        # Screen offsets from the center, rolled into the planet's frame
        sx = px[:, None] + ox[None, :] - cx
        sy = py[:, None] + oy[None, :] - cy
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        rx = sx * cos_r + sy * sin_r
        ry = -sx * sin_r + sy * cos_r
        light = unit(light)
        light_r = np.array([light[0] * cos_r + light[1] * sin_r, -light[0] * sin_r + light[1] * cos_r, light[2]])

        x = rx / radius
        y = ry / (radius * oblate)
        on_disc = x * x + y * y < 1
        coverage = on_disc.mean(axis=1)

        # Rings lie in the equatorial plane, seen from `tilt` above it: a ring
        # point w ahead of the planet's center (toward us) shows tilt-scaled
        # below it, and covers the planet only when w > 0
        ring_front = np.zeros(len(px), dtype=np.float32)
        ring_back = np.zeros(len(px), dtype=np.float32)
        ring_color = np.zeros((len(px), 3), dtype=np.float32)
        if rings:
            w = ry / math.sin(tilt)
            r = np.hypot(rx, w) / radius
            density = np.zeros_like(r)
            color = np.zeros(r.shape + (3,), dtype=np.float32)
            for inner, outer, c, alpha in rings:
                inside = (r >= inner) & (r < outer)
                density = np.where(inside, alpha, density)
                color[inside] = c
            ring_front = (density * (w > 0)).mean(axis=1)
            ring_back = (density * ((w <= 0) & ~on_disc)).mean(axis=1)
            weight = density.sum(axis=1, keepdims=True)
            ring_color = (color * density[..., None]).sum(axis=1) / np.maximum(weight, 1e-6) * 0.9

        # The planet's surface, sampled at each pixel's center (pulled onto
        # the rim for edge pixels whose center is just outside)
        center = SUPERSAMPLE * SUPERSAMPLE // 2
        x0 = x[:, center]
        y0 = y[:, center]
        rim = np.hypot(x0, y0)
        pull = np.where(rim > 0.999, 0.999 / np.maximum(rim, 1e-6), 1.0)
        x0, y0 = x0 * pull, y0 * pull
        z0 = np.sqrt(np.clip(1 - x0 * x0 - y0 * y0, 0, 1))
        cos_t, sin_t = math.cos(tilt), math.sin(tilt)
        by = y0 * cos_t - z0 * sin_t  # pole tipped toward us by tilt
        bz = y0 * sin_t + z0 * cos_t
        lat = np.arcsin(np.clip(-by, -1, 1))
        lon = np.arctan2(x0, bz)
        row = np.clip(((math.pi / 2 - lat) / math.pi * TEX_H).astype(np.int32), 0, TEX_H - 1)
        lit = np.clip(x0 * light_r[0] + y0 * light_r[1] + z0 * light_r[2], 0, 1)
        brightness = (ambient + (1 - ambient) * lit ** 0.8) * (1 - limb * (1 - z0))

        keep = coverage > 0
        self.index = (py * width + px)[keep]
        self.row_base = row[keep] * TEX_W
        self.column = ((lon / (2 * math.pi) + 0.5) * TEX_W)[keep].astype(np.float32)
        self.weight = (brightness * coverage)[keep, None].astype(np.float32)
        self.night = (coverage * np.clip(1 - lit * 8, 0, 1))[keep, None].astype(np.float32)
        self.cloud_shade = (np.clip(lit, 0.04, 1) * coverage)[keep, None].astype(np.float32)

        # Static layers: what the planet, back rings and glow hide of the stars,
        # and what the back rings and atmosphere add
        n = width * height
        self.occlusion = np.zeros(n, dtype=np.float32)
        self.static = np.zeros((n, 3), dtype=np.float32)
        flat = py * width + px
        self.occlusion[flat] = np.clip(coverage + ring_back, 0, 1)
        self.static[flat] = ring_color * ring_back[:, None]
        if atmosphere is not None:
            # A thin glow just outside the limb, brightest on the lit side,
            # averaged over the samples so it stays smooth
            d = np.hypot(x, y)
            facing = np.clip((x * light_r[0] + y * light_r[1]) / np.maximum(d, 1e-6) + 0.35, 0, 1)
            glow = (np.clip(1 - (d - 1) / 0.2, 0, 1) ** 2 * ~on_disc * facing).mean(axis=1)
            self.static[flat] += np.asarray(atmosphere, dtype=np.float32) * glow[:, None]
        front = ring_front > 0
        self.ring_index = flat[front]
        self.ring_keep = (1 - ring_front[front])[:, None]
        self.ring_add = (ring_color * ring_front[:, None])[front]

    def sample(self, values: np.ndarray, column: np.ndarray) -> np.ndarray:
        """values (texture-shaped, flattened) at each pixel's row and the
        given fractional column, blended between neighboring columns."""
        c0 = column.astype(np.int32)
        f = column - c0
        i0 = self.row_base + c0
        i1 = self.row_base + (c0 + 1) % TEX_W
        a = values[i0]
        if values.ndim == 2:
            f = f[:, None]
        return a + (values[i1] - a) * f

    def draw(self, out: np.ndarray, t: float) -> None:
        """Add the planet (and front rings) at time t seconds to out, a flat
        (pixels, 3) float32 frame already holding the background."""
        turn = self.phase + self.spin * t + self.libration * math.sin(t * 2 * math.pi / 30)
        column = (self.column - turn) % TEX_W  # minus: the surface moves west to east
        color = self.sample(self.texture, column) * self.weight
        if self.clouds is not None:
            cover = self.sample(self.clouds, (self.column - turn * 1.2) % TEX_W)[:, None]
            color += (255 * self.cloud_shade - color) * cover
        if self.lights is not None:
            color += self.lights[self.row_base + column.astype(np.int32)] * self.night
        out[self.index] += color
        if len(self.ring_index):
            out[self.ring_index] = out[self.ring_index] * self.ring_keep + self.ring_add


def moon_light() -> list[float]:
    """Sunlight on the Moon for its current phase: from behind it at new
    moon, from the right at first quarter, from the front at full."""
    phase = ((time.time() - NEW_MOON_EPOCH) / 86400 / SYNODIC_DAYS) % 1.0
    angle = 2 * math.pi * phase
    return [math.sin(angle), -0.1, -math.cos(angle)]


SUN = [-0.75, -0.25, 0.6]  # from the upper left, a little in front

BODIES = [
    lambda w, h: Planet(w, h, earth_surface, radius=14, spin_seconds=50, light=SUN, tilt=0.3,
                        roll=-0.4, ambient=0.02, atmosphere=(40, 110, 255)),
    lambda w, h: Planet(w, h, moon_surface, radius=14, spin_seconds=0, light=moon_light(),
                        ambient=0.09, libration=2.5),
    lambda w, h: Planet(w, h, mars_surface, radius=13, spin_seconds=45, light=SUN, tilt=0.35,
                        roll=0.3, ambient=0.02, atmosphere=(160, 70, 45)),
    lambda w, h: Planet(w, h, jupiter_surface, radius=14.5, spin_seconds=28, light=SUN, tilt=0.05,
                        roll=0.05, oblate=0.93, ambient=0.02, limb=0.35),
    lambda w, h: Planet(w, h, saturn_surface, radius=9, spin_seconds=30, light=SUN, tilt=0.45,
                        roll=0.25, oblate=0.9, ambient=0.02, limb=0.3, rings=SATURN_RINGS),
]


class Planets:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        n = width * height
        self.star_index = np.array(random.sample(range(n), min(STAR_COUNT, n)))
        self.star_color = np.array(
            [np.array(random.choice(STAR_TINTS)) * random.uniform(0.25, 0.8) for _ in self.star_index],
            dtype=np.float32,
        )
        self.star_speed = np.random.uniform(0.5, 2.0, len(self.star_index)).astype(np.float32)
        self.star_phase = np.random.uniform(0, 2 * math.pi, len(self.star_index)).astype(np.float32)
        self.body = random.randrange(len(BODIES))
        self.planet = BODIES[self.body](width, height)
        self.frames = 0  # frames this planet has been shown
        self.total_frames = 0

    def frame(self) -> np.ndarray:
        planet_frames = round(PLANET_SECONDS / FRAME_SECONDS)
        if self.frames >= planet_frames:
            # Next body, built while the panel is black between fades
            self.body = (self.body + 1) % len(BODIES)
            self.planet = BODIES[self.body](self.width, self.height)
            self.frames = 0
        t = self.frames * FRAME_SECONDS
        self.frames += 1
        self.total_frames += 1

        out = self.planet.static.copy()
        twinkle = 0.7 + 0.3 * np.sin(self.total_frames * FRAME_SECONDS * self.star_speed + self.star_phase)
        hidden = 1 - self.planet.occlusion[self.star_index]
        out[self.star_index] += self.star_color * (twinkle * hidden)[:, None]
        self.planet.draw(out, t)

        fade = min(1.0, t / FADE_SECONDS, (PLANET_SECONDS - t) / FADE_SECONDS)
        if fade < 1:
            out *= max(fade, 0.0) ** 2
        np.minimum(out, 255, out=out)
        return out.astype(np.uint8).reshape(-1)


SCENE = Planets


def main() -> None:
    run_scene(Planets, "Planets LED matrix display")


if __name__ == "__main__":
    main()
