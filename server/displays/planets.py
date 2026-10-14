#!/usr/bin/env python
"""Planets: a slowly rotating planet, lit from one side, against twinkling
stars. Cycles through Earth (clouds, city lights on the night side), the Moon
(in its current phase), Mars, Jupiter (Great Red Spot) and Saturn (rings
passing in front of and behind it, shadows between them), fading each planet
out and the next in while the stars carry on.

Each pixel's spot on the sphere, its lighting and its anti-aliased edge are
worked out once per planet; a frame then just slides the planet's surface
texture past those spots, averaging the texture over each pixel's footprint
so the rotation glides, even near the limb where the surface is foreshortened.

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

# Surfaces are equirectangular textures of cells: columns run east from
# longitude -180, rows south from latitude 90
TEX_W, TEX_H = 128, 64
SUPERSAMPLE = 4  # samples per pixel per axis, for anti-aliased edges

STAR_COUNT = 40
STAR_TINTS = [(255, 255, 255), (190, 210, 255), (255, 230, 190)]

# Earth's land in 10-degree cells: rows 90N..90S, columns 180W..180E.
# "#" is land, "d" desert.
WORLD = [
    "............####....................",
    "......##########...#...############.",
    ".########.#.###.#.##################",
    "....#####.###....###############.#..",
    ".....########....######dd##dd####...",
    "......d###.......#...dddd#d#####....",
    "......dd........ddddddddd#####......",
    "........##......######d..#.##.#.....",
    "..........###....######.....##......",
    "..........#####....###......##.##...",
    "..........####.....####.......###...",
    "...........###.....dd#.......dddd#..",
    "...........##......##........#dd#..#",
    "...........#....................#..#",
    "...........#........................",
    "...........##.....##################",
    "..##########...####################.",
    "####################################",
]

# Synodic month and a known new moon (2000-01-06 18:14 UTC), for the phase
SYNODIC_DAYS = 29.530588853
NEW_MOON_EPOCH = 947182440
# Near new moon the real Moon is lost in the Sun's glare; show at least this
# much of a month's crescent (about 3 days' worth) instead of a dark disc
MIN_CRESCENT = 0.1


# --- Texture helpers ---------------------------------------------------------


def upsample(grid: np.ndarray, w: int = TEX_W, h: int = TEX_H) -> np.ndarray:
    """Smoothly interpolate a coarse (rows, cols) grid of cells up to (h, w)
    cells, wrapping around in longitude."""
    gh, gw = grid.shape
    # Each fine cell's center, in coarse cells from the first coarse center
    x = (np.arange(w) + 0.5) * gw / w - 0.5
    y = np.clip((np.arange(h) + 0.5) * gh / h - 0.5, 0, gh - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.minimum(np.floor(y).astype(int), gh - 2)
    fx = x - x0
    fy = y - y0
    fx = (fx * fx * (3 - 2 * fx))[None, :]
    fy = (fy * fy * (3 - 2 * fy))[:, None]
    x1 = (x0 + 1) % gw
    x0 %= gw
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
    """Latitude in degrees of each texture row's center, as a column vector."""
    return (90 - (np.arange(TEX_H, dtype=np.float32) + 0.5) * 180 / TEX_H)[:, None]


def longitudes() -> np.ndarray:
    return ((np.arange(TEX_W, dtype=np.float32) + 0.5) * 360 / TEX_W - 180)[None, :]


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


def running_sums(texture: np.ndarray) -> np.ndarray:
    """A (TEX_H, TEX_W, channels) texture as a flat lookup table for averaging
    along its rows: entry row * 3 * TEX_W + c holds the sum of the row's
    texels before column c, then texel c itself. The row is laid out three
    times over, so spans can start anywhere in the first two copies (as the
    surface turns) and run on into the next without wrapping."""
    tiled = np.concatenate([texture] * 3, axis=1).astype(np.float64)
    before = np.cumsum(tiled, axis=1) - tiled
    return np.concatenate([before, tiled], axis=-1).reshape(TEX_H * 3 * TEX_W, -1).astype(np.float32)


# --- Surfaces: (texture, extras) for each body --------------------------------


def earth_surface(rng):
    lat, lon = latitudes(), longitudes()
    world = np.array([[ch != "." for ch in row] for row in WORLD], dtype=np.float32)
    dry = np.array([[ch == "d" for ch in row] for row in WORLD], dtype=np.float32)
    # Noise roughs up the coasts, which would otherwise follow the 10-degree
    # cells in rounded steps
    shape = upsample(world) + (fbm(rng, base=6) - 0.5) * 1.2
    land = shape > 0.5

    detail = fbm(rng, base=8)
    desert = np.clip((upsample(dry) + (detail - 0.5) * 1.2 - 0.4) * 4, 0, 1)
    ground = mix(mix((25, 105, 35), (80, 130, 45), detail), (205, 165, 100), desert)
    ground = mix(ground, (70, 90, 55), np.clip((abs(lat) - 52) / 10, 0, 1))  # boreal forest
    polar = lat + (detail - 0.5) * 20
    ice = (polar > 72) | (polar < -56) | ((lat > 60) & (lon > -55) & (lon < -20))  # and Greenland
    ground = np.where(ice[..., None], np.float32(230), ground)
    ocean = mix((0, 22, 80), (0, 60, 140), np.clip(shape * 3 - 0.5, 0, 1))  # paler in the shallows
    texture = np.where(land[..., None], ground, ocean)

    # City lights: sparse warm dots on land, most at mid northern latitudes.
    # Each is over-bright, as a pixel shows the average of the texels it
    # covers (more of them toward the limb).
    density = np.where((lat > 15) & (lat < 58), 0.16, 0.05) * (1 - desert)
    lights = (land & ~ice & (rng.random((TEX_H, TEX_W)) < density)).astype(np.float32)
    lights = lights[..., None] * np.array([500, 370, 190], dtype=np.float32)

    # Clouds streak east-west and gather along the equator and in the
    # storm tracks around 60 degrees, leaving the subtropics and deserts clear
    clouds = fbm(rng, base=8, octaves=4, stretch=2.0) + 0.07 * np.cos(np.radians(lat) * 6)
    clouds = np.clip((clouds - 0.55) * 4.5, 0, 0.85) * (1 - 0.7 * desert)
    return texture, {"lights": lights, "clouds": clouds}


def moon_surface(rng):
    lat, lon = latitudes(), longitudes()
    base = mix((150, 150, 145), (185, 185, 180), fbm(rng, base=10))
    # Dark maria, mostly on the side facing Earth (longitude near 0)
    near = np.clip(np.cos(np.radians(lon)) * 1.4, 0, 1)
    maria = np.clip((fbm(rng, base=3) - 0.48) * 6, 0, 1) * near * np.clip(1 - abs(lat) / 70, 0, 1)
    texture = mix(base, (85, 85, 92), maria * 0.85)
    # Craters: darker floors, bright rims
    y, x = np.mgrid[0:TEX_H, 0:TEX_W] + 0.5
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
    # Dark markings, commonest in a band just south of the equator
    albedo = fbm(rng, base=3, octaves=5) + 0.1 * np.exp(-(((lat + 15) / 20) ** 2))
    texture = mix((205, 110, 60), (175, 80, 45), np.clip(albedo * 1.5 - 0.3, 0, 1))
    texture = mix(texture, (95, 45, 30), np.clip((albedo - 0.55) * 3.5, 0, 1))
    caps = abs(lat) + (fbm(rng, base=8) - 0.5) * 30 > 75
    return np.where(caps[..., None], np.float32(240), texture), {}


def jupiter_surface(rng):
    lat, lon = latitudes(), longitudes()
    # Turbulence: bend the bands with noise stretched east-west
    wobble = (fbm(rng, base=6, octaves=4, stretch=4.0) - 0.5) * 9
    # Pale zones and dark belts; the north and south equatorial belts are the
    # broad dark ones either side of the equator
    stops = [
        (90, (130, 120, 112)), (52, (170, 152, 128)), (36, (215, 200, 170)), (29, (185, 135, 92)),
        (23, (232, 218, 185)), (17, (175, 100, 62)), (9, (182, 110, 70)), (5, (236, 222, 192)),
        (-5, (228, 208, 172)), (-9, (172, 98, 60)), (-17, (180, 110, 68)), (-21, (232, 214, 180)),
        (-25, (225, 208, 172)), (-30, (185, 135, 92)), (-38, (215, 198, 168)), (-55, (170, 152, 128)),
        (-90, (130, 120, 112)),
    ]
    texture = bands(stops, lat + wobble)
    texture *= (0.9 + 0.2 * fbm(rng, base=8, octaves=3, stretch=6.0))[..., None]  # streaks
    # The Great Red Spot, with a pale collar
    d = np.hypot(lon / 12, (lat + 22) / 7.5)
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
SHADOW = 0.85  # how much light the planet's shadow on the rings takes away


# --- Rendering ---------------------------------------------------------------


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def ring_at(r: np.ndarray, rings) -> tuple[np.ndarray, np.ndarray]:
    """Ring opacity and color at distances r (in planet radii) from the
    center of the ring plane."""
    density = np.zeros(r.shape, dtype=np.float32)
    color = np.zeros(r.shape + (3,), dtype=np.float32)
    for inner, outer, c, alpha in rings:
        inside = (r >= inner) & (r < outer)
        density[inside] = alpha
        color[inside] = c
    return density, color


class Planet:
    """One body, precomputed for drawing at the middle of the panel.

    Directions are in screen terms: x right, y down, z toward the viewer.
    radius is a fraction of the panel's height (or half its width, if less).
    tilt tips the north pole toward the viewer, then roll turns the planet
    clockwise. A dusty body (the Moon) is lit evenly out to its limb, as
    airless dusty surfaces are, rather than shading off toward it."""

    def __init__(self, width: int, height: int, surface, *, radius: float, spin_seconds: float,
                 light, tilt: float = 0.0, roll: float = 0.0, oblate: float = 1.0,
                 ambient: float = 0.03, limb: float = 0.0, dusty: bool = False, atmosphere=None,
                 rings=None, libration: float = 0.0, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        texture, extras = surface(rng)
        self.lights = extras.get("lights") is not None
        if self.lights:
            # Night lights ride along in the texture, after the day colors
            texture = np.concatenate([texture, extras["lights"]], axis=-1)
        self.table = running_sums(texture)
        clouds = extras.get("clouds")
        self.clouds = None if clouds is None else running_sums(clouds[..., None])
        self.spin = TEX_W / spin_seconds if spin_seconds else 0.0  # texture columns per second
        self.libration = libration
        self.phase = random.uniform(0, TEX_W)  # start facing a random longitude

        # Every pixel that the planet, its rings or its glow can touch, with
        # SUPERSAMPLE x SUPERSAMPLE sample points each
        radius *= min(width / 2, height)
        reach = radius * (rings[-1][1] if rings else 1.2) + 1
        cx, cy = width / 2, height / 2
        xs = np.arange(max(0, int(cx - reach)), min(width, int(cx + reach) + 1))
        ys = np.arange(max(0, int(cy - reach)), min(height, int(cy + reach) + 1))
        py, px = (a.reshape(-1) for a in np.meshgrid(ys, xs, indexing="ij"))
        sub = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE
        oy, ox = (a.reshape(-1) for a in np.meshgrid(sub, sub, indexing="ij"))
        # Screen offsets from the center, the samples' and then each pixel's
        # center's, rolled into the planet's frame
        cos_r, sin_r = math.cos(roll), math.sin(roll)
        sx = np.concatenate([px[:, None] + ox[None, :], px[:, None] + 0.5], axis=1) - cx
        sy = np.concatenate([py[:, None] + oy[None, :], py[:, None] + 0.5], axis=1) - cy
        rx = sx * cos_r + sy * sin_r
        ry = -sx * sin_r + sy * cos_r
        light = unit(light)
        light_r = np.array([light[0] * cos_r + light[1] * sin_r, -light[0] * sin_r + light[1] * cos_r, light[2]])

        # Planet radii, with y stretched to make an oblate planet round
        x = rx / radius
        y = ry / (radius * oblate)
        on_disc = (x * x + y * y < 1)[:, :-1]
        coverage = on_disc.mean(axis=1)

        # Where each sample (pulled onto the rim if just outside it) lands on
        # the surface: depth z, and latitude and longitude once the pole is
        # tipped toward us by tilt
        rim = np.hypot(x, y)
        pull = np.where(rim > 0.999, 0.999 / np.maximum(rim, 1e-6), 1.0)
        x, y = x * pull, y * pull
        z = np.sqrt(np.clip(1 - x * x - y * y, 0, 1))
        cos_t, sin_t = math.cos(tilt), math.sin(tilt)
        lat = np.arcsin(np.clip(z * sin_t - y * cos_t, -1, 1))
        lon = np.arctan2(x, y * sin_t + z * cos_t)
        # Each pixel shows the texture at its center's row, averaged over the
        # span of columns its samples reach (scaled up from the samples'
        # spread to the pixel's)
        spread = (lon[:, :-1] - lon[:, -1:] + math.pi) % (2 * math.pi) - math.pi
        span = np.ptp(spread, axis=1) * TEX_W / (2 * math.pi) * SUPERSAMPLE / (SUPERSAMPLE - 1)
        span = np.clip(span, 1, TEX_W - 1)
        column = (lon[:, -1] / (2 * math.pi) + 0.5) * TEX_W
        row = np.clip(((math.pi / 2 - lat[:, -1]) / math.pi * TEX_H).astype(np.int32), 0, TEX_H - 1)

        # Sunlight at each pixel's center
        xc, yc, zc = x[:, -1], y[:, -1], z[:, -1]
        lit = np.clip(xc * light_r[0] + yc * light_r[1] + zc * light_r[2], 0, 1)
        if dusty:
            lit = np.clip(2 * lit / np.maximum(lit + zc, 1e-6), 0, 1)  # Lommel-Seeliger
        sun = lit ** 0.8

        # Rings lie in the equatorial plane, seen from `tilt` above it: a ring
        # point w ahead of the planet's center (toward us) shows tilt-scaled
        # below it, and covers the planet only when w > 0
        ring_front = np.zeros(len(px), dtype=np.float32)
        ring_back = np.zeros(len(px), dtype=np.float32)
        ring_color = np.zeros((len(px), 3), dtype=np.float32)
        if rings:
            # The rings' shadow on the planet: follow each sample on the disc
            # toward the Sun to the ring plane (in radii, unstretched)
            x, y, z = x[:, :-1], y[:, :-1] * oblate, z[:, :-1]
            toward = light_r[1] * cos_t - light_r[2] * sin_t  # the light along the plane's normal
            ahead = -(y * cos_t - z * sin_t) / toward
            hit_w = (y + ahead * light_r[1]) * sin_t + (z + ahead * light_r[2]) * cos_t
            blocked, _ = ring_at(np.hypot(x + ahead * light_r[0], hit_w), rings)
            clear = ((1 - blocked * (ahead > 0)) * on_disc).sum(axis=1)
            sun *= clear / np.maximum(on_disc.sum(axis=1), 1)

            w = ry[:, :-1] / sin_t
            r = np.hypot(rx[:, :-1], w) / radius
            density, color = ring_at(r, rings)
            # The planet's shadow on the rings: ring points whose path to the
            # Sun passes through the planet (in its stretched, round frame)
            point = np.stack([rx[:, :-1], ry[:, :-1] / oblate, w * cos_t], axis=-1) / radius
            light_s = unit([light_r[0], light_r[1] / oblate, light_r[2]])
            along = point @ light_s
            shaded = (along < 0) & ((point * point).sum(axis=-1) - along * along < 1)
            color *= (1 - SHADOW * shaded)[..., None]

            ring_front = (density * (w > 0)).mean(axis=1)
            ring_back = (density * ((w <= 0) & ~on_disc)).mean(axis=1)
            weight = density.sum(axis=1, keepdims=True)
            ring_color = (color * density[..., None]).sum(axis=1) / np.maximum(weight, 1e-6) * 0.9
        brightness = (ambient + (1 - ambient) * sun) * (1 - limb * (1 - zc))

        keep = coverage > 0
        self.index = (py * width + px)[keep]
        self.row_base = row[keep] * 3 * TEX_W
        self.start = ((column - span / 2) % TEX_W)[keep].astype(np.float32)
        self.span = span[keep].astype(np.float32)
        self.inv_span = (1 / span)[keep, None].astype(np.float32)
        self.weight = (brightness * coverage)[keep, None].astype(np.float32)
        self.night = (coverage * np.clip(1 - lit * 8, 0, 1))[keep, None].astype(np.float32)
        self.cloud_color = (255 * np.clip(lit, 0.04, 1) * coverage)[keep, None].astype(np.float32)

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
            x, y = rx[:, :-1] / radius, ry[:, :-1] / (radius * oblate)
            d = np.hypot(x, y)
            facing = np.clip((x * light_r[0] + y * light_r[1]) / np.maximum(d, 1e-6) + 0.35, 0, 1)
            glow = (np.clip(1 - (d - 1) / 0.2, 0, 1) ** 2 * ~on_disc * facing).mean(axis=1)
            self.static[flat] += np.asarray(atmosphere, dtype=np.float32) * glow[:, None]
        front = ring_front > 0
        self.ring_index = flat[front]
        self.ring_keep = (1 - ring_front[front])[:, None]
        self.ring_add = (ring_color * ring_front[:, None])[front]

    def running_sum(self, table: np.ndarray, column: np.ndarray) -> np.ndarray:
        """table's running sum along each pixel's row up to a fractional
        column (0 <= column < 3 * TEX_W)."""
        part, whole = np.modf(column)
        entry = table.take(self.row_base + whole.astype(np.int32), axis=0)
        k = entry.shape[1] // 2
        return entry[:, :k] + entry[:, k:] * part[:, None]

    def sample(self, table: np.ndarray, turn: float) -> np.ndarray:
        """The texture in table (see running_sums) averaged over each pixel's
        span of columns, with the surface turned turn columns east."""
        # Wrapped in float64, so start stays in the table's first two copies
        start = self.start + (TEX_W - turn % TEX_W)
        return (self.running_sum(table, start + self.span) - self.running_sum(table, start)) * self.inv_span

    def draw(self, out: np.ndarray, t: float) -> None:
        """Add the planet (and front rings) at time t seconds to out, a flat
        (pixels, 3) float32 frame already holding the background."""
        turn = self.phase + self.spin * t + self.libration * math.sin(t * 2 * math.pi / 30)
        texel = self.sample(self.table, turn)
        color = texel[:, :3] * self.weight
        if self.lights:
            color += texel[:, 3:] * self.night
        if self.clouds is not None:
            # Clouds drift a little faster than the ground, veiling the lights
            color += (self.cloud_color - color) * self.sample(self.clouds, turn * 1.2)
        out[self.index] += color
        if len(self.ring_index):
            out[self.ring_index] = out[self.ring_index] * self.ring_keep + self.ring_add


def moon(width: int, height: int) -> Planet:
    """The Moon in its current phase: lit from behind it at new moon, from
    the right at first quarter, from the front at full. Earthshine on its
    night side is strongest when it's a crescent, as Earth is then near full
    seen from the Moon."""
    phase = ((time.time() - NEW_MOON_EPOCH) / 86400 / SYNODIC_DAYS) % 1.0
    angle = 2 * math.pi * min(max(phase, MIN_CRESCENT), 1 - MIN_CRESCENT)
    earthshine = 0.03 + 0.1 * (1 + math.cos(angle)) / 2
    return Planet(width, height, moon_surface, radius=0.44, spin_seconds=0,
                  light=[math.sin(angle), -0.1, -math.cos(angle)], ambient=earthshine, dusty=True,
                  libration=2.5)


SUN = [-0.75, -0.25, 0.6]  # from the upper left, a little in front
# Seen from Earth the outer planets are lit nearly from the front; a little
# off it, Saturn's shadow shows as a notch in the rings behind it
OUTER_SUN = [-0.5, -0.35, 0.8]

BODIES = [
    lambda w, h: Planet(w, h, earth_surface, radius=0.44, spin_seconds=50, light=SUN, tilt=0.3,
                        roll=-0.4, ambient=0.02, atmosphere=(40, 110, 255)),
    moon,
    lambda w, h: Planet(w, h, mars_surface, radius=0.41, spin_seconds=45, light=SUN, tilt=0.35,
                        roll=0.3, ambient=0.02, atmosphere=(160, 70, 45)),
    lambda w, h: Planet(w, h, jupiter_surface, radius=0.45, spin_seconds=28, light=OUTER_SUN, tilt=0.05,
                        roll=0.05, oblate=0.93, ambient=0.02, limb=0.35),
    lambda w, h: Planet(w, h, saturn_surface, radius=0.28, spin_seconds=30, light=OUTER_SUN, tilt=0.45,
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
        # float64, as the twinkle's phase grows without bound (see matrix_common)
        self.star_speed = np.random.uniform(0.5, 2.0, len(self.star_index))
        self.star_phase = np.random.uniform(0, 2 * math.pi, len(self.star_index))
        self.body = random.randrange(len(BODIES))
        self.show(BODIES[self.body](width, height))
        self.total_frames = 0

    def show(self, planet: Planet) -> None:
        self.planet = planet
        self.star_shown = 1 - planet.occlusion[self.star_index, None]
        self.frames = 0  # frames this planet has been shown

    def frame(self) -> np.ndarray:
        if self.frames >= round(PLANET_SECONDS / FRAME_SECONDS):
            # Next body, built while only the stars are showing
            self.body = (self.body + 1) % len(BODIES)
            self.show(BODIES[self.body](self.width, self.height))
        t = self.frames * FRAME_SECONDS
        self.frames += 1
        self.total_frames += 1

        twinkle = 0.7 + 0.3 * np.sin(self.total_frames * FRAME_SECONDS * self.star_speed + self.star_phase)
        stars = self.star_color * twinkle[:, None]
        out = self.planet.static.copy()
        out[self.star_index] += stars * self.star_shown
        self.planet.draw(out, t)

        fade = min(1.0, t / FADE_SECONDS, (PLANET_SECONDS - t) / FADE_SECONDS)
        if fade < 1:
            # Fade from the bare starfield to the full frame
            sky = np.zeros_like(out)
            sky[self.star_index] = stars
            out = sky + (out - sky) * max(fade, 0.0) ** 2
        np.minimum(out, 255, out=out)
        return out.astype(np.uint8).reshape(-1)


SCENE = Planets


def main() -> None:
    run_scene(Planets, "Planets LED matrix display")


if __name__ == "__main__":
    main()
