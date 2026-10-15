#!/usr/bin/env python
"""Koi pond: koi seen from above, gliding over a pebbly bottom under lily
pads, with sunlight rippling across the floor.

Each koi is a chain of points that follows its head, so its body bends as it
turns, drawn as soft overlapping discs (wider at the shoulders, tapering to
the tail) with its tail sweeping side to side, its fins sculling, and its
shadow on the bottom below. Koi swim at different depths, the deeper ones
dimmer and bluer, slide under the lily pads, and now and then rise to kiss
the surface. They come in the classic varieties: kohaku, sanke, showa,
tancho, ogon, platinum, asagi, chagoi and kumonryu.

Every minute or so something happens (see EVENTS): food is sprinkled and the
koi crowd in to gulp it, a rain shower dimples the water, a dragonfly darts
about and lands on a pad, a heron's shadow sweeps over and the koi bolt for
cover, a frog climbs onto a pad, croaks and jumps in, or leaves (blossom
petals in spring) drift down and float away. AQUARIUM_EVENT=<name> shows
just that one (see aquarium_common):

    AQUARIUM_EVENT=feeding uv run server/displays/preview.py koi_pond --seconds 30

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random
from datetime import datetime

import numpy as np

from aquarium_common import FRAME_SECONDS, Event, Tank, frames_for
from matrix_common import run_scene

WIDTH, HEIGHT = 64, 32
PIXELS = WIDTH * HEIGHT
TAU = math.tau

# The pond is bigger than the panel: koi wander anywhere in POND (left, top,
# right, bottom), now and then out of view, but mostly within VIEW
POND = (-14.0, -10.0, WIDTH + 14.0, HEIGHT + 10.0)
VIEW = (5.0, 4.0, WIDTH - 5.0, HEIGHT - 4.0)
OUT_OF_VIEW = 0.15  # chance a new destination is out of view

# Colors
WATER = (6, 40, 44)  # deep koi fade toward this
FLOOR = (40, 46, 34)
PEBBLES = [(72, 64, 50), (56, 60, 52), (88, 78, 62), (62, 52, 42), (46, 54, 40)]
WATER_TINT = 0.45  # how much the water tints the floor
CAUSTIC_LOW, CAUSTIC_HIGH = 0.8, 0.75  # floor brightness: base, plus up to this in the light
PAD_DARK = (24, 78, 30)
PAD_LIGHT = (66, 128, 50)
PETAL = (245, 150, 185)
PETAL_TIP = (255, 205, 220)
LOTUS_HEART = (250, 210, 70)
GLINT = (70, 75, 65)

# Koi colors
WHITE = (240, 238, 228)
RED = (228, 60, 26)
BLACK = (26, 26, 32)
GOLD = (248, 172, 36)
PLATINUM = (214, 220, 226)
ASAGI_BLUE = (92, 120, 152)
ASAGI_RED = (232, 104, 44)
BRONZE = (150, 102, 58)

# Ripples spread at RIPPLE_SPEED pixels a frame for up to RIPPLE_FRAMES
# (raindrops' fade sooner)
RIPPLE_SPEED = 0.22
RIPPLE_FRAMES = 46
RAINDROP_FRAMES = 26

# Shadows fall down and to the right of whatever casts them
SHADOW_DX, SHADOW_DY = 0.8, 1.0
SHADOW_MAX = 0.6

# The lily pads: (x, y, radius), and which one has the lotus flower
PADS = [(6.0, 6.0, 4.6), (13.5, 2.5, 3.0), (54.0, 24.0, 5.0), (61.5, 14.0, 3.2), (37.0, 30.0, 2.8)]
LOTUS_PAD = 2


def mix(a, b, t: float) -> tuple:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def wrap(angle: float) -> float:
    """An angle in -pi..pi."""
    return (angle + math.pi) % TAU - math.pi


YY, XX = (a.astype(np.float32) for a in np.mgrid[0:HEIGHT, 0:WIDTH])
WATER_RGB = np.array(WATER, np.float32)
SHIMMER = np.array((255, 255, 240), np.float32)


# --- Drawing -----------------------------------------------------------------

# Every disc is drawn over the 7x7 pixels around it, so radii up to 2
DISC_DY, DISC_DX = (a.ravel() for a in np.mgrid[-3:4, -3:4])


class Discs:
    """A batch of soft round blobs at sub-pixel positions, drawn together as
    one layer: where they overlap, their colors blend. Creatures here are
    built out of these, so they can bend and turn any way."""

    def __init__(self) -> None:
        self.x: list[float] = []
        self.y: list[float] = []
        self.r: list[float] = []
        self.w: list[float] = []
        self.c: list[tuple] = []
        self.batches: list[tuple] = []  # (x, y, r, w, colors) arrays from add_many

    def add(self, x: float, y: float, r: float, weight: float = 1.0, color=(0, 0, 0)) -> None:
        self.x.append(x)
        self.y.append(y)
        self.r.append(r)
        self.w.append(weight)
        self.c.append(color)

    def add_many(self, x: np.ndarray, y: np.ndarray, r: np.ndarray, weight, colors: np.ndarray) -> None:
        """Many discs at once, as arrays (weight may be one number)."""
        n = len(x)
        self.batches.append((x, y, r, np.broadcast_to(np.float32(weight), (n,)) if np.ndim(weight) == 0 else weight,
                             colors))

    def arrays(self) -> tuple:
        parts = list(self.batches)
        if self.x:
            parts.append((np.array(self.x, np.float32), np.array(self.y, np.float32), np.array(self.r, np.float32),
                          np.array(self.w, np.float32), np.array(self.c, np.float32)))
        if len(parts) == 1:
            return parts[0]
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(5))

    def __bool__(self) -> bool:
        return bool(self.x or self.batches)

    def raster(self) -> tuple:
        """(row, column, coverage, which disc) for every pixel a disc touches,
        and the discs' colors."""
        x, y, r, w, colors = self.arrays()
        x = np.asarray(x, np.float32)[:, None]
        y = np.asarray(y, np.float32)[:, None]
        xi = np.floor(x).astype(np.int32) + DISC_DX
        yi = np.floor(y).astype(np.int32) + DISC_DY
        d = np.sqrt((xi - x) ** 2 + (yi - y) ** 2)
        cover = np.clip(np.asarray(r, np.float32)[:, None] + 0.5 - d, 0, 1) * np.asarray(w, np.float32)[:, None]
        ok = (xi >= 0) & (xi < WIDTH) & (yi >= 0) & (yi < HEIGHT) & (cover > 0.004)
        return yi[ok], xi[ok], cover[ok], np.nonzero(ok)[0], colors

    def paint(self, frame: np.ndarray) -> None:
        """Paint the blobs over a (height, width, 3) frame."""
        if not self:
            return
        yi, xi, cover, which, colors = self.raster()
        if not len(cover):
            return
        # Work in the box the blobs cover
        x0, x1, y0, y1 = xi.min(), xi.max() + 1, yi.min(), yi.max() + 1
        w, h = x1 - x0, y1 - y0
        local = (yi - y0) * w + (xi - x0)
        alpha = np.zeros(w * h, np.float32)
        np.maximum.at(alpha, local, cover)
        total = np.maximum(np.bincount(local, cover, w * h), 1e-6)
        rgb = np.bincount((local[:, None] * 3 + RGB_INDEX).ravel(),
                          (cover[:, None] * np.asarray(colors, np.float32)[which]).ravel(), w * h * 3)
        blend = rgb.reshape(h, w, 3) / total.reshape(h, w, 1)
        region = frame[y0:y1, x0:x1]
        region += (blend - region) * alpha.reshape(h, w, 1)

    def shade(self) -> np.ndarray:
        """The blobs' summed weights per pixel (height, width), for a shadow."""
        if not self:
            return np.zeros((HEIGHT, WIDTH), np.float32)
        yi, xi, cover, _, _ = self.raster()
        return np.bincount(yi * WIDTH + xi, cover, PIXELS).astype(np.float32).reshape(HEIGHT, WIDTH)


RGB_INDEX = np.arange(3)


def make_rings() -> list[tuple[np.ndarray, np.ndarray]]:
    """A ripple's ring at each age, centered in a square: how much it
    brightens (crest) or darkens (trough) the water as a multiplier, and the
    light it adds, each at full strength."""
    rings = []
    for age in range(RIPPLE_FRAMES):
        r = age * RIPPLE_SPEED
        reach = math.ceil(r + 2.5)
        yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1].astype(np.float32)
        d = np.hypot(xx, yy)
        crest = np.exp(-((d - r) / 0.65) ** 2)
        trough = np.exp(-((d - r + 1.2) / 0.65) ** 2)
        rings.append(((0.55 * crest - 0.3 * trough)[..., None], (crest * 26)[..., None]))
    return rings


RINGS = make_rings()


# --- Koi ---------------------------------------------------------------------


def pattern(variety: str, n: int) -> tuple[list[tuple], tuple, bool]:
    """A koi's colors head to tail, its fins' color, and whether it shimmers."""
    def patches(base, color, count, lengths):
        colors = [base] * n
        for _ in range(count):
            size = random.randint(*lengths)
            start = random.randint(0, max(0, n - 3 - size))
            for i in range(start, min(n - 2, start + size)):
                colors[i] = color
        return colors

    if variety == "kohaku":
        return patches(WHITE, RED, random.randint(2, 3), (2, 4)), mix(WHITE, RED, 0.15), False
    if variety == "sanke":
        colors = patches(WHITE, RED, 2, (2, 3))
        for i in random.sample(range(2, n - 2), 2):
            colors[i] = BLACK
        return colors, WHITE, False
    if variety == "showa":
        colors = patches(BLACK, RED, 2, (2, 4))
        colors[random.randint(3, n - 4)] = WHITE
        colors[0] = colors[1] = random.choice((RED, BLACK))
        return colors, mix(BLACK, WHITE, 0.4), False
    if variety == "tancho":
        colors = [WHITE] * n
        colors[1] = RED
        return colors, WHITE, False
    if variety == "ogon":
        return [GOLD] * n, mix(GOLD, WHITE, 0.3), True
    if variety == "platinum":
        return [PLATINUM] * n, WHITE, True
    if variety == "asagi":
        colors = [ASAGI_BLUE] * n
        colors[0] = mix(ASAGI_BLUE, WHITE, 0.5)
        colors[-3] = mix(ASAGI_BLUE, ASAGI_RED, 0.5)
        return colors, ASAGI_RED, False
    if variety == "chagoi":
        return [BRONZE] * n, mix(BRONZE, WHITE, 0.25), False
    return patches(BLACK, WHITE, 3, (1, 3)), mix(BLACK, WHITE, 0.5), False  # kumonryu


VARIETIES = ["kohaku", "sanke", "showa", "tancho", "ogon", "platinum", "asagi", "chagoi", "kumonryu"]
KOI = 7


class Koi:
    """A koi: its head swims about, steering toward where it's going, and the
    rest of its body follows, a pixel between each point.

    Events steer koi by giving them a task (feed, flee, hide), a target, a
    pace and a depth to swim at (z_to: 0 at the surface to 1 deep)."""

    def __init__(self, pond: "Pond", variety: str) -> None:
        self.pond = pond
        self.variety = variety
        self.length = random.uniform(13.5, 16.0) if variety == "chagoi" else random.uniform(10.0, 14.0)
        self.n = round(self.length)
        self.colors, self.fin_color, self.shimmer = pattern(variety, self.n)
        girth = self.length * random.uniform(0.14, 0.16)
        self.radii = [girth * self.profile(i / (self.n - 1)) for i in range(self.n)]
        # The same as arrays, for drawing
        self.u = np.linspace(0, 1, self.n, dtype=np.float32)
        self.index = np.arange(self.n, dtype=np.float32)
        self.sweep_shape = self.u ** 1.6
        self.radius_array = np.array(self.radii, np.float32)
        self.color_array = np.array(self.colors, np.float32)
        self.fin_at = max(1, round(0.24 * (self.n - 1)))
        self.heading = random.uniform(0, TAU)
        self.x, self.y = random.uniform(*VIEW[::2]), random.uniform(*VIEW[1::2])
        self.points = [(self.x - i * math.cos(self.heading), self.y - i * math.sin(self.heading))
                       for i in range(self.n)]
        self.cruise = random.uniform(0.09, 0.16) * (0.85 if variety == "chagoi" else 1.0)
        self.speed = self.cruise
        self.pace = 1.0
        self.z = self.z_to = random.uniform(0.15, 0.75)
        self.phase = random.uniform(0, TAU)
        self.task: str | None = None
        self.calm_in = 0  # frames until a hiding koi comes out again
        self.pause = 0
        self.gulp = 0
        self.wander()

    @staticmethod
    def profile(u: float) -> float:
        """Body width along its length (0 head, 1 tail): a rounded head,
        broad shoulders, tapering to the tail."""
        if u < 0.25:
            return 0.72 + 0.28 * math.sin(u / 0.25 * math.pi / 2)
        return 1.0 - 0.68 * ((u - 0.25) / 0.75) ** 1.3

    def wander(self) -> None:
        """Pick somewhere new to swim to, usually in view."""
        area = POND if random.random() < OUT_OF_VIEW else VIEW
        self.target = (random.uniform(area[0], area[2]), random.uniform(area[1], area[3]))
        if random.random() < 0.3:
            self.z_to = random.uniform(0.1, 0.8)

    def swim(self, others: list["Koi"]) -> None:
        tx, ty = self.target
        dx, dy = tx - self.x, ty - self.y
        distance = math.hypot(dx, dy)
        want = self.cruise * self.pace
        if self.task is None:
            if distance < 3:
                self.wander()
            if self.pause:
                self.pause -= 1
                want = self.cruise * 0.12
            elif random.random() < 0.0015:
                self.pause = random.randint(40, 160)
            # Now and then a koi near the surface comes up for a kiss
            if self.z < 0.3 and random.random() < 0.0012:
                self.pond.ripple(*self.mouth(), 0.35)
        elif self.task == "hide":
            want = self.cruise * 0.25 if distance > 2 else 0.01
            self.calm_in -= 1
            if self.calm_in <= 0:
                self.task, self.pace = None, 1.0
                self.wander()
        else:
            # Slow down to arrive rather than overshoot
            want *= min(1.0, 0.3 + distance / 6)
        desired = math.atan2(dy, dx)

        # Steer around koi swimming at about the same depth
        for other in others:
            if other is self or abs(other.z - self.z) > 0.35:
                continue
            for px, py in other.points[::3]:
                ox, oy = px - self.x, py - self.y
                if ox * ox + oy * oy < 12 and ox * math.cos(self.heading) + oy * math.sin(self.heading) > 0:
                    side = math.cos(self.heading) * oy - math.sin(self.heading) * ox
                    desired = self.heading - 0.9 * (1 if side > 0 else -1)
                    break

        max_turn = 0.02 + 0.12 * self.speed
        self.heading = wrap(self.heading + max(-max_turn, min(max_turn, wrap(desired - self.heading) * 0.08)))
        self.speed += (want - self.speed) * (0.08 if self.task == "flee" else 0.04)
        self.x += math.cos(self.heading) * self.speed
        self.y += math.sin(self.heading) * self.speed
        self.z += (self.z_to - self.z) * (0.03 if self.task == "flee" else 0.01)
        self.phase = (self.phase + 0.07 + self.speed * 1.3) % TAU
        if self.gulp:
            self.gulp -= 1

        # The body follows the head, a pixel between points
        points = self.points
        points[0] = (self.x, self.y)
        for i in range(1, self.n):
            px, py = points[i - 1]
            qx, qy = points[i]
            ddx, ddy = qx - px, qy - py
            d = math.hypot(ddx, ddy) or 1.0
            points[i] = (px + ddx / d, py + ddy / d)

    def mouth(self) -> tuple[float, float]:
        return self.x + math.cos(self.heading) * 0.8, self.y + math.sin(self.heading) * 0.8

    def visible(self) -> bool:
        return any(-3 < x < WIDTH + 3 and -3 < y < HEIGHT + 3 for x, y in self.points[::3])

    def draw(self, t: float, shadow: Discs) -> Discs:
        """The koi as discs, and its shadow (added to shadow)."""
        z = self.z
        fade = 0.12 + 0.5 * z
        alpha = 1.0 - 0.25 * z
        p = np.array(self.points, np.float32)
        # Each point's direction toward the head, and across the body
        ahead = np.empty_like(p)
        ahead[1:] = p[:-1] - p[1:]
        ahead[0] = (math.cos(self.heading), math.sin(self.heading))
        ahead /= np.maximum(np.hypot(ahead[:, 0], ahead[:, 1]), 1e-6)[:, None]
        across = np.stack([-ahead[:, 1], ahead[:, 0]], axis=1)
        # The tail sweeps side to side, more the faster it swims, most at the tip
        sweep = 0.3 + min(self.speed, 0.4) * 2.2
        side = sweep * self.sweep_shape * np.sin(self.phase - self.u * 3.2)
        spine = p + across * side[:, None]
        colors = self.color_array
        if self.shimmer:
            glint = 0.25 * np.maximum(0.0, np.sin(t * 1.6 - self.index * 0.7))[:, None]
            colors = colors + (SHIMMER - colors) * glint
        body = Discs()
        body.add_many(spine[:, 0], spine[:, 1], self.radius_array, alpha, colors + (WATER_RGB - colors) * fade)
        # Nearer the bottom, the shadow is closer and darker
        offset = 0.6 + (1 - z) * 2.0
        shadow.add_many(spine[:, 0] + offset * SHADOW_DX, spine[:, 1] + offset * SHADOW_DY,
                        self.radius_array + 0.3, 0.2 + 0.12 * z, colors)

        fin = mix(self.fin_color, WATER, fade + 0.15)
        # Pectoral fins, sculling
        i = self.fin_at
        (x, y), (ax, ay), (nx, ny) = spine[i], ahead[i], across[i]
        reach = self.radii[i] + 0.55 + 0.25 * math.sin(self.phase * 0.5)
        for s in (1, -1):
            body.add(x + nx * reach * s - ax * 0.35, y + ny * reach * s - ay * 0.35, 0.6, 0.5 * alpha, fin)
        # The tail fin, a fan behind the last point
        (x, y), (ax, ay), (nx, ny) = spine[-1], ahead[-1], across[-1]
        for s in (1, -1):
            spread = 0.7 + 0.25 * math.sin(self.phase + s)
            body.add(x - ax * 0.9 + nx * spread * s, y - ay * 0.9 + ny * spread * s, 0.65, 0.55 * alpha, fin)
        if self.gulp:
            mx, my = self.mouth()
            body.add(mx, my, 0.5, 0.8, (40, 22, 22))
        return body


# --- The pond ----------------------------------------------------------------


class PondEvent(Event):
    """An event in the pond. Besides step() and end(), it can add to the
    shadows on the floor, draw under the lily pads (in the water) and above
    them (on the surface or in the air)."""

    def shadows(self, shadow: Discs) -> None:
        pass

    def under(self, frame: np.ndarray, t: float) -> None:
        pass

    def above(self, frame: np.ndarray, t: float) -> None:
        pass


class Pond(Tank):
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the pond is drawn for a {WIDTH}x{HEIGHT} panel")
        self.width, self.height = width, height
        self.floor = height
        self.fish = []  # (the Tank's fish; the koi swim differently)
        varieties = random.sample(VARIETIES, KOI - 1) + [random.choice(("kohaku", "ogon", "showa"))]
        self.koi = [Koi(self, v) for v in varieties]
        self.ripples: list[list] = []  # [x, y, age, strength, life]
        self.glints: list[list[float]] = []  # [x, y, age, life]
        self.frame_count = 0
        self.light = self.light_to = 1.0  # sunlight, dimmed by rain
        self.caustics = self.caustics_to = 1.0
        self.make_floor()
        self.make_pads()
        self.init_events()

    # Setting ----------------------------------------------------------

    def make_floor(self) -> None:
        rng = random.Random(11)
        floor = np.empty((PIXELS, 3), np.float32)
        floor[:] = FLOOR
        stones = Discs()
        for _ in range(110):
            stones.add(rng.uniform(-1, WIDTH), rng.uniform(-1, HEIGHT), rng.uniform(0.5, 1.4),
                       rng.uniform(0.5, 0.95), rng.choice(PEBBLES))
        floor = floor.reshape(HEIGHT, WIDTH, 3)
        stones.paint(floor)
        # Deeper toward the middle
        edge = np.minimum(np.minimum(XX, WIDTH - 1 - XX) / 18, np.minimum(YY, HEIGHT - 1 - YY) / 10)
        floor *= (1.05 - 0.25 * np.clip(edge, 0, 1))[..., None]
        self.floor_rgb = floor * (1 - WATER_TINT) + np.array(WATER, np.float32) * WATER_TINT

    def make_pads(self) -> None:
        """The lily pads (and lotus) as a layer over the water, and their
        shadow on the floor."""
        alpha = np.zeros((HEIGHT, WIDTH), np.float32)
        color = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        shadow = np.zeros((HEIGHT, WIDTH), np.float32)
        rng = random.Random(5)
        for cx, cy, r in PADS:
            notch = rng.uniform(0, TAU)
            for dx, dy, layer in ((0.0, 0.0, None), (SHADOW_DX * 2, SHADOW_DY * 2, shadow)):
                d = np.hypot(XX - cx - dx, YY - cy - dy)
                angle = np.arctan2(YY - cy - dy, XX - cx - dx)
                gap = np.abs((angle - notch + np.pi) % TAU - np.pi)
                cover = np.clip(r + 0.5 - d, 0, 1) * np.clip((gap - 0.3) * np.maximum(d, 0.5) + 0.5, 0, 1)
                if layer is not None:
                    np.maximum(layer, cover * 0.45, out=layer)
                    continue
                # Lighter toward the rim, with faint veins from the middle
                shade = 0.45 + 0.55 * np.clip(d / r, 0, 1)
                veins = np.cos(angle * 7 + notch) > 0.9
                tone = np.where(veins, shade * 0.75, shade)[..., None]
                pad = np.array(PAD_DARK, np.float32) + (np.array(PAD_LIGHT, np.float32) - PAD_DARK) * tone
                color += (pad - color) * cover[..., None]
                np.maximum(alpha, cover, out=alpha)
        lotus = Discs()
        cx, cy, r = PADS[LOTUS_PAD]
        lx, ly = cx - 0.8, cy - 0.8
        for k in range(6):
            a = k * TAU / 6 + 0.3
            lotus.add(lx + math.cos(a) * 1.3, ly + math.sin(a) * 1.3, 0.85, 1.0, PETAL)
            lotus.add(lx + math.cos(a) * 1.8, ly + math.sin(a) * 1.8, 0.4, 0.8, PETAL_TIP)
        lotus.add(lx, ly, 0.7, 1.0, LOTUS_HEART)
        lotus.paint(color)
        self.pad_color = color
        self.pad_color = self.pad_color.reshape(HEIGHT, WIDTH, 3)
        self.pad_alpha = alpha[..., None]
        self.pad_shadow = shadow

    def on_pad(self, x: float, y: float, margin: float = 0.0) -> bool:
        return any(math.hypot(x - cx, y - cy) < r - margin for cx, cy, r in PADS)

    def ripple(self, x: float, y: float, strength: float = 0.6, life: int = RIPPLE_FRAMES) -> None:
        if -10 < x < WIDTH + 10 and -10 < y < HEIGHT + 10 and not self.on_pad(x, y):
            self.ripples.append([x, y, 0, strength, life])

    # Drawing ----------------------------------------------------------

    def caustic_light(self, t: float) -> np.ndarray:
        """Sunlight focused by the moving surface into a net of bright lines
        on the floor, 0-1."""
        # (Phases wrapped in float64 before meeting float32 arrays, so they
        # stay smooth after weeks)
        a = np.sin(XX * 0.52 + (t * 0.8) % TAU + np.sin(YY * 0.37 + (t * 0.45) % TAU) * 1.6)
        b = np.sin(YY * 0.61 - (t * 0.65) % TAU + np.sin(XX * 0.29 - (t * 0.38) % TAU) * 1.6)
        return np.clip(1 - np.abs(a + b) * 1.4, 0, 1) ** 2

    def draw_ripples(self, frame: np.ndarray) -> None:
        live = []
        for ripple in self.ripples:
            x, y, age, strength, life = ripple
            ripple[2] += 1
            if age >= life:
                continue
            live.append(ripple)
            fade = strength * (1 - age / life) ** 1.5
            dim, glow = RINGS[int(age)]
            reach = dim.shape[0] // 2
            cx, cy = round(x), round(y)
            x0, y0 = max(0, cx - reach), max(0, cy - reach)
            x1, y1 = min(WIDTH, cx + reach + 1), min(HEIGHT, cy + reach + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            kx, ky = x0 - (cx - reach), y0 - (cy - reach)
            region = frame[y0:y1, x0:x1]
            region *= 1 + dim[ky:ky + y1 - y0, kx:kx + x1 - x0] * fade
            region += glow[ky:ky + y1 - y0, kx:kx + x1 - x0] * fade
        self.ripples = live

    def draw_glints(self, frame: np.ndarray) -> None:
        """Sun sparkling here and there on the surface."""
        if random.random() < 0.12 * self.light ** 4:
            self.glints.append([random.uniform(0, WIDTH - 1), random.uniform(0, HEIGHT - 1), 0, random.randint(4, 9)])
        live = []
        for glint in self.glints:
            x, y, age, life = glint
            glint[2] += 1
            if age < life:
                live.append(glint)
                c, r = round(x), round(y)
                if 0 <= c < WIDTH and 0 <= r < HEIGHT:
                    frame[r, c] += np.array(GLINT, np.float32) * math.sin(math.pi * age / life)
        self.glints = live

    def frame(self) -> np.ndarray:
        t = self.frame_count * FRAME_SECONDS
        self.direct()
        self.light += (self.light_to - self.light) * 0.02
        self.caustics += (self.caustics_to - self.caustics) * 0.02
        for koi in self.koi:
            koi.swim(self.koi)

        caustic = self.caustic_light(t) * (CAUSTIC_HIGH * self.caustics)
        frame = self.floor_rgb * (self.light * (CAUSTIC_LOW + caustic))[..., None]

        # Shadows on the floor, then the koi from the deepest up
        shadow = Discs()
        bodies = [(koi.z, koi.draw(t, shadow)) for koi in self.koi if koi.visible()]
        if self.event:
            self.event.shadows(shadow)
        shade = np.minimum(shadow.shade() + self.pad_shadow, SHADOW_MAX)
        frame *= (1 - shade * self.light)[..., None]
        for z, body in sorted(bodies, key=lambda b: -b[0]):
            body.paint(frame)
        if self.event:
            self.event.under(frame, t)
        self.draw_ripples(frame)

        # The lily pads, then whatever's on or above the water
        frame += (self.pad_color * self.light - frame) * self.pad_alpha
        if self.event:
            self.event.above(frame, t)
        self.draw_glints(frame)
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


# --- Events ------------------------------------------------------------------

PELLET = (176, 124, 66)


class Feeding(PondEvent):
    """A handful of food lands on the water, and the koi notice one by one,
    crowd in and gulp it down at the surface."""

    def __init__(self, pond: Pond) -> None:
        super().__init__(pond)
        sx, sy = random.uniform(16, 48), random.uniform(9, 23)
        self.pellets = []
        for _ in range(random.randint(10, 15)):
            a, d = random.uniform(0, TAU), abs(random.gauss(0, 2.4))
            self.pellets.append({"x": sx + math.cos(a) * d, "y": sy + math.sin(a) * d,
                                 "vx": math.cos(a) * random.uniform(0, 0.02), "vy": math.sin(a) * random.uniform(0, 0.02),
                                 "at": random.randint(0, frames_for(1.5)), "eaten": False})
        self.notice = {id(koi): frames_for(random.uniform(0.8, 4.0)) for koi in pond.koi}
        self.frame_count = 0
        self.done_at = None
        self.limit = frames_for(50)

    def live(self) -> list[dict]:
        return [p for p in self.pellets if p["at"] <= self.frame_count and not p["eaten"]]

    def step(self) -> bool:
        self.frame_count += 1
        for p in self.pellets:
            if p["at"] == self.frame_count:
                self.tank.ripple(p["x"], p["y"], 0.3)
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.995
            p["vy"] *= 0.995
        live = self.live()
        for koi in self.tank.koi:
            if self.frame_count < self.notice[id(koi)] or not live:
                continue
            mx, my = koi.mouth()
            nearest = min(live, key=lambda p: (p["x"] - mx) ** 2 + (p["y"] - my) ** 2)
            koi.task, koi.pace, koi.z_to = "feed", 2.2, 0.0
            koi.target = (nearest["x"], nearest["y"])
            if math.hypot(nearest["x"] - mx, nearest["y"] - my) < 1.3 and koi.z < 0.35:
                nearest["eaten"] = True
                koi.gulp = 8
                self.tank.ripple(nearest["x"], nearest["y"], 0.7)
                live.remove(nearest)
        if not live and self.frame_count > max(p["at"] for p in self.pellets):
            self.done_at = self.done_at or self.frame_count
            for koi in self.tank.koi:
                if koi.task == "feed":
                    koi.task, koi.pace = None, 1.0
                    koi.wander()
        return self.frame_count < self.limit and (self.done_at is None or self.frame_count < self.done_at + frames_for(2))

    def end(self) -> None:
        for koi in self.tank.koi:
            if koi.task == "feed":
                koi.task, koi.pace = None, 1.0
                koi.z_to = random.uniform(0.1, 0.7)
                koi.wander()

    def shadows(self, shadow: Discs) -> None:
        for p in self.live():
            shadow.add(p["x"] + 2.5 * SHADOW_DX, p["y"] + 2.5 * SHADOW_DY, 0.4, 0.3)

    def above(self, frame: np.ndarray, t: float) -> None:
        food = Discs()
        for p in self.live():
            bob = 0.15 * math.sin(t * 3 + p["at"])
            food.add(p["x"], p["y"] + bob, 0.45, 1.0, PELLET)
        food.paint(frame)


class Rain(PondEvent):
    """A shower: drops dimple the water all over, the light dims and the koi
    sink a little deeper until it passes."""

    def __init__(self, pond: Pond) -> None:
        super().__init__(pond)
        self.frames = frames_for(random.uniform(25, 40))
        self.ramp = frames_for(5)
        self.frame_count = 0
        for koi in pond.koi:
            if koi.task is None:
                koi.z_to = min(1.0, koi.z_to + 0.35)
                koi.pace = 0.8

    def step(self) -> bool:
        self.frame_count += 1
        f = self.frame_count
        heavy = max(0.0, min(1.0, f / self.ramp, (self.frames - f) / self.ramp))
        self.tank.light_to = 1 - 0.25 * heavy
        self.tank.caustics_to = 1 - 0.85 * heavy
        drops = heavy * 0.9
        while drops > 0:
            if random.random() < drops:
                self.tank.ripple(random.uniform(-2, WIDTH + 2), random.uniform(-2, HEIGHT + 2),
                                 random.uniform(0.5, 0.9), RAINDROP_FRAMES)
            drops -= 1
        return f < self.frames

    def end(self) -> None:
        self.tank.light_to = self.tank.caustics_to = 1.0
        for koi in self.tank.koi:
            if koi.task is None:
                koi.pace = 1.0
                koi.z_to = random.uniform(0.1, 0.7)


DRAGONFLY_BODY = (46, 128, 236)
DRAGONFLY_BAND = (24, 70, 150)
DRAGONFLY_EYES = (40, 190, 140)
WING = (205, 225, 245)


class Dragonfly(PondEvent):
    """A dragonfly darts in, hovers here and there, dips to touch the water,
    rests a while on a lily pad, and darts off again."""

    def __init__(self, pond: Pond) -> None:
        super().__init__(pond)
        side = random.choice((-1, 1))
        self.x, self.y = (-6.0 if side > 0 else WIDTH + 6.0), random.uniform(4, HEIGHT - 4)
        self.heading = 0.0 if side > 0 else math.pi
        self.height = 1.0  # above the water: 1 flying, 0 touching
        # Its plan: a few stops, a dip or two, a rest on a pad, then away
        stops = [("fly", self.spot()) for _ in range(random.randint(2, 4))]
        for i in random.sample(range(len(stops)), min(2, len(stops))):
            stops.insert(i + 1, ("dip", None))
        cx, cy, r = random.choice(PADS[:LOTUS_PAD] + PADS[LOTUS_PAD + 1:])
        rest = (cx + random.uniform(-0.4, 0.4) * r, cy + random.uniform(-0.4, 0.4) * r)
        stops += [("fly", rest), ("rest", None), ("fly", self.spot()),
                  ("fly", (random.choice((-8.0, WIDTH + 8.0)), random.uniform(0, HEIGHT)))]
        self.plan = stops
        self.doing = None
        self.wait = 0

    @staticmethod
    def spot() -> tuple[float, float]:
        return random.uniform(8, WIDTH - 8), random.uniform(5, HEIGHT - 5)

    def step(self) -> bool:
        if self.wait:
            self.wait -= 1
            if self.doing == "dip":
                # Down to touch the water, and back up
                self.height = abs(2 * self.wait / frames_for(0.6) - 1)
                if self.wait == frames_for(0.3):
                    self.tank.ripple(self.x, self.y, 0.45)
            elif self.doing == "hover":
                self.x += random.uniform(-0.08, 0.08)
                self.y += random.uniform(-0.08, 0.08)
            return True
        if self.doing == "fly":
            tx, ty = self.goal
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            if d > 0.8:
                speed = min(1.1, 0.15 + d * 0.12)
                self.heading = math.atan2(dy, dx)
                self.x += dx / d * speed
                self.y += dy / d * speed
                self.height += (1.0 - self.height) * 0.2
                return True
            if self.plan and self.plan[0][0] == "fly":
                self.doing = "hover"
                self.wait = frames_for(random.uniform(0.6, 1.8))
                return True
        if not self.plan:
            return False
        self.doing, goal = self.plan.pop(0)
        if self.doing == "fly":
            self.goal = goal
        elif self.doing == "dip":
            self.wait = frames_for(0.6)
        elif self.doing == "rest":
            self.height = 0.0
            self.wait = frames_for(random.uniform(4, 8))
        return True

    def creature(self, wings: float) -> tuple[Discs, Discs]:
        """The dragonfly's wings and body as discs."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        nx, ny = -s, c
        wing = Discs()
        for along in (0.7, -0.2):
            for side in (1, -1):
                for k in range(1, 4):
                    w = k * 0.95
                    wing.add(self.x + c * (along - 0.15 * k) + nx * w * side,
                             self.y + s * (along - 0.15 * k) + ny * w * side, 0.5, wings, WING)
        body = Discs()
        body.add(self.x + c * 1.4, self.y + s * 1.4, 0.55, 1.0, DRAGONFLY_EYES)
        body.add(self.x + c * 0.5, self.y + s * 0.5, 0.5, 1.0, DRAGONFLY_BODY)
        for k in range(5):
            along = -0.5 - k * 0.8
            body.add(self.x + c * along, self.y + s * along, 0.38, 1.0,
                     DRAGONFLY_BAND if k % 2 else DRAGONFLY_BODY)
        return wing, body

    def shadows(self, shadow: Discs) -> None:
        lift = 0.6 + self.height * 4.0
        wing, body = self.creature(0.3)
        for discs in (wing, body):
            for x, y, r, w in zip(discs.x, discs.y, discs.r, discs.w):
                shadow.add(x + lift * SHADOW_DX, y + lift * SHADOW_DY, r, 0.18)

    def above(self, frame: np.ndarray, t: float) -> None:
        resting = self.doing == "rest" and self.wait
        wings = 0.6 if resting else random.uniform(0.25, 0.6)  # a blur in flight
        wing, body = self.creature(wings)
        wing.paint(frame)
        body.paint(frame)


class Heron(PondEvent):
    """A heron's shadow sweeps over the pond, and the koi bolt for cover deep
    under the lily pads, coming out again a while after it's gone."""

    def __init__(self, pond: Pond) -> None:
        super().__init__(pond)
        # Across the pond on a slant
        y0 = random.uniform(-4, HEIGHT + 4)
        y1 = random.uniform(-4, HEIGHT + 4)
        if random.random() < 0.5:
            (x0, x1) = (-22.0, WIDTH + 22.0)
        else:
            (x0, x1) = (WIDTH + 22.0, -22.0)
        self.start, self.finish = (x0, y0), (x1, y1)
        self.frames = frames_for(random.uniform(4.5, 6.5))
        self.heading = math.atan2(y1 - y0, x1 - x0)
        self.frame_count = 0
        self.scared = False

    def position(self) -> tuple[float, float]:
        k = self.frame_count / self.frames
        return (self.start[0] + (self.finish[0] - self.start[0]) * k,
                self.start[1] + (self.finish[1] - self.start[1]) * k)

    def step(self) -> bool:
        self.frame_count += 1
        x, y = self.position()
        if not self.scared and -12 < x < WIDTH + 12:
            self.scared = True
            for koi in self.tank.koi:
                # To the nearest pad, deep
                cx, cy, r = min(PADS, key=lambda p: math.hypot(p[0] - koi.x, p[1] - koi.y))
                koi.task, koi.pace, koi.z_to = "flee", 4.0, 1.0
                koi.target = (cx + random.uniform(-1, 1), cy + random.uniform(-1, 1))
        return self.frame_count < self.frames

    def end(self) -> None:
        for koi in self.tank.koi:
            if koi.task == "flee":
                koi.task, koi.pace = "hide", 1.0
                koi.calm_in = frames_for(random.uniform(4, 12))

    def shape(self) -> Discs:
        x, y = self.position()
        c, s = math.cos(self.heading), math.sin(self.heading)
        nx, ny = -s, c
        bird = Discs()
        for along, r in ((3, 1.2), (1.5, 1.7), (0, 1.9), (-1.5, 1.6), (-3, 1.1)):
            bird.add(x + c * along, y + s * along, r, 1.0)
        for along in (4.5, 5.5, 6.5):
            bird.add(x + c * along, y + s * along, 0.6, 1.0)  # the neck, tucked
        bird.add(x + c * 7.4, y + s * 7.4, 0.8, 1.0)
        bird.add(x + c * 8.8, y + s * 8.8, 0.4, 1.0)  # the bill
        for along in (-4.5, -6, -7.5):
            bird.add(x + c * along, y + s * along, 0.35, 1.0)  # legs trailing
        for side in (1, -1):
            for k in range(1, 13):
                sweep = -0.18 * k
                bird.add(x + c * sweep + nx * k * side, y + s * sweep + ny * k * side, 1.9 - 0.1 * k, 1.0)
        return bird

    def above(self, frame: np.ndarray, t: float) -> None:
        frame *= (1 - np.minimum(self.shape().shade(), 1.0) * 0.62)[..., None]


FROG = (92, 104, 34)  # olive, to stand out on a green pad
FROG_BACK = (150, 190, 60)
FROG_EYES = (245, 210, 60)
FROG_THROAT = (225, 220, 170)


class Frog(PondEvent):
    """A frog swims in under the water, climbs onto a lily pad, sits and
    croaks a while, then jumps in with a splash and swims off."""

    def __init__(self, pond: Pond) -> None:
        super().__init__(pond)
        # A pad well in view (the lotus pad has room beside the flower)
        self.pad = random.choice([p for p in PADS if 4 < p[0] < WIDTH - 4 and 4 < p[1] < HEIGHT - 4])
        cx, cy, r = self.pad
        # In from the nearest edge, to the pad's edge facing the middle
        towards = math.atan2(HEIGHT / 2 - cy, WIDTH / 2 - cx)
        self.seat = (cx + math.cos(towards) * r * 0.35, cy + math.sin(towards) * r * 0.35)
        self.landing = (cx + math.cos(towards) * (r + 1.5), cy + math.sin(towards) * (r + 1.5))
        start_angle = towards + random.uniform(-0.6, 0.6)
        self.x, self.y = cx + math.cos(start_angle) * 40, cy + math.sin(start_angle) * 40
        self.heading = math.atan2(self.landing[1] - self.y, self.landing[0] - self.x)
        self.phase = "swim"
        self.kick = 0.0
        self.wait = 0
        self.lift = 0.0
        self.croak = 0.0
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        if self.phase in ("swim", "leave"):
            tx, ty = self.landing if self.phase == "swim" else self.away
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            self.kick = (self.kick + 0.06) % 1.0
            speed = 0.05 + 0.35 * max(0.0, math.sin(self.kick * TAU))  # a push, then a glide
            self.heading += wrap(math.atan2(dy, dx) - self.heading) * 0.1
            self.x += math.cos(self.heading) * speed
            self.y += math.sin(self.heading) * speed
            if self.phase == "swim" and d < 1.0:
                self.phase = "climb"
                self.tank.ripple(self.x, self.y, 0.35)
            if self.phase == "leave":
                return -8 < self.x < WIDTH + 8 and -8 < self.y < HEIGHT + 8
        elif self.phase == "climb":
            tx, ty = self.seat
            self.x += (tx - self.x) * 0.08
            self.y += (ty - self.y) * 0.08
            if math.hypot(tx - self.x, ty - self.y) < 0.2:
                self.phase = "sit"
                self.wait = frames_for(random.uniform(6, 12))
                # Turn to face the water to jump into
                cx, cy, r = self.pad
                self.jump_to = self.spot_beyond()
                self.croaks = sorted(random.sample(range(self.wait), 3))
        elif self.phase == "sit":
            self.wait -= 1
            if self.croaks and self.wait <= self.croaks[-1]:
                self.croaks.pop()
                self.croak = 1.0
            self.croak = max(0.0, self.croak - 0.03)
            want = math.atan2(self.jump_to[1] - self.y, self.jump_to[0] - self.x)
            if self.wait < frames_for(1.5):
                self.heading += wrap(want - self.heading) * 0.15
            if self.wait <= 0:
                self.phase = "jump"
                self.jump_from = (self.x, self.y)
                self.wait = frames_for(0.5)
                self.jump_frames = self.wait
        elif self.phase == "jump":
            self.wait -= 1
            k = 1 - self.wait / self.jump_frames
            (x0, y0), (x1, y1) = self.jump_from, self.jump_to
            self.x, self.y = x0 + (x1 - x0) * k, y0 + (y1 - y0) * k
            self.lift = math.sin(math.pi * k)
            if self.wait <= 0:
                self.lift = 0.0
                self.tank.ripple(self.x, self.y, 1.1)
                self.tank.ripple(self.x, self.y, 0.5)
                self.phase = "leave"
                # Off the nearest way out
                self.away = (self.x + math.cos(self.heading) * 60, self.y + math.sin(self.heading) * 60)
        return True

    def spot_beyond(self) -> tuple[float, float]:
        """Open water a jump away from the pad."""
        cx, cy, r = self.pad
        for _ in range(30):
            a = random.uniform(0, TAU)
            d = r + random.uniform(4, 8)
            x, y = cx + math.cos(a) * d, cy + math.sin(a) * d
            if 4 < x < WIDTH - 4 and 4 < y < HEIGHT - 4 and not self.tank.on_pad(x, y):
                return x, y
        return WIDTH / 2, HEIGHT / 2

    def creature(self, wet: bool) -> Discs:
        c, s = math.cos(self.heading), math.sin(self.heading)
        nx, ny = -s, c
        x, y = self.x, self.y
        fade = 0.4 if wet else 0.0
        alpha = 0.8 if wet else 1.0
        size = 1.2 + 0.3 * self.lift

        def at(along, side, r, color, w=1.0):
            frog.add(x + (c * along + nx * side) * size, y + (s * along + ny * side) * size, r * size,
                     w * alpha, mix(color, WATER, fade))

        frog = Discs()
        swimming = self.phase in ("swim", "leave")
        stretch = swimming and math.sin(self.kick * TAU) > 0
        jumping = self.phase == "jump"
        for side in (1, -1):
            if stretch or jumping:
                at(-2.4, 0.9 * side, 0.45, FROG)
                at(-3.4, 0.7 * side, 0.4, FROG)
            else:
                at(-1.5, 1.4 * side, 0.55, FROG)
                at(-0.5, 1.7 * side, 0.45, FROG)
            at(1.0, 1.2 * side, 0.35, FROG)
        at(-0.6, 0, 1.2, FROG)
        at(0.4, 0, 1.25, FROG)
        at(-0.3, 0, 0.5, FROG_BACK, 0.7)
        at(1.4, 0, 0.95, FROG)
        if self.croak > 0:
            at(1.9, 0, 0.4 + 0.6 * self.croak, FROG_THROAT, 0.9)
        for side in (1, -1):
            at(1.5, 0.75 * side, 0.4, FROG_EYES)
        return frog

    def shadows(self, shadow: Discs) -> None:
        if self.phase == "jump":
            frog = self.creature(False)
            lift = 1.0 + self.lift * 4.0
            for x, y, r in zip(frog.x, frog.y, frog.r):
                shadow.add(x + lift * SHADOW_DX, y + lift * SHADOW_DY, r, 0.25)

    def under(self, frame: np.ndarray, t: float) -> None:
        if self.phase in ("swim", "leave"):
            self.creature(True).paint(frame)

    def above(self, frame: np.ndarray, t: float) -> None:
        if self.phase in ("climb", "sit", "jump"):
            self.creature(False).paint(frame)


LEAF_COLORS = [(232, 112, 30), (214, 60, 32), (240, 178, 40), (196, 84, 36)]
PETAL_COLORS = [(250, 178, 198), (252, 200, 214), (244, 160, 186)]


class Leaves(PondEvent):
    """Leaves (blossom petals in spring) flutter down onto the water, each
    landing with a ripple, and drift slowly away on the breeze."""

    def __init__(self, pond: Pond) -> None:
        super().__init__(pond)
        self.petals = 3 <= datetime.now().month <= 5
        count = random.randint(9, 15) if self.petals else random.randint(4, 7)
        wind = random.uniform(0.02, 0.045) * random.choice((1, -1))
        self.leaves = []
        for _ in range(count):
            self.leaves.append({
                "x": random.uniform(4, WIDTH - 4), "y": random.uniform(3, HEIGHT - 3),
                "vx": wind * random.uniform(0.8, 1.2), "vy": random.uniform(-0.008, 0.008),
                "spin": random.uniform(0, TAU), "turn": random.uniform(-0.01, 0.01),
                "land": random.randint(frames_for(0.5), frames_for(12)), "fall": frames_for(1.6),
                "color": random.choice(PETAL_COLORS if self.petals else LEAF_COLORS)})
        self.frame_count = 0
        self.limit = frames_for(80)

    def step(self) -> bool:
        self.frame_count += 1
        f = self.frame_count
        for leaf in self.leaves:
            if f == leaf["land"]:
                self.tank.ripple(leaf["x"], leaf["y"], 0.45)
            if f >= leaf["land"]:
                leaf["x"] += leaf["vx"]
                leaf["y"] += leaf["vy"] + 0.004 * math.sin(f * 0.02 + leaf["spin"])
                leaf["spin"] += leaf["turn"]
        gone = all(f > leaf["land"] and not (-4 < leaf["x"] < WIDTH + 4) for leaf in self.leaves)
        return not gone and f < self.limit

    def falling(self, leaf: dict) -> float:
        """How far a leaf still has to fall: 1 just appeared, 0 on the water."""
        start = leaf["land"] - leaf["fall"]
        if self.frame_count < start:
            return -1.0
        return max(0.0, (leaf["land"] - self.frame_count) / leaf["fall"])

    def shape(self, leaf: dict, discs: Discs, dx: float = 0.0, dy: float = 0.0, weight=None) -> None:
        x, y, a = leaf["x"] + dx, leaf["y"] + dy, leaf["spin"]
        # Fade out at the end, if any are still about
        fade = min(1.0, (self.limit - self.frame_count) / frames_for(3))
        if self.petals:
            discs.add(x, y, 0.55, (weight or 0.95) * fade, leaf["color"])
            discs.add(x + math.cos(a) * 0.6, y + math.sin(a) * 0.6, 0.35, (weight or 0.8) * fade, leaf["color"])
            return
        discs.add(x, y, 0.75, (weight or 1.0) * fade, leaf["color"])
        for k in range(5):
            b = a + k * TAU / 5
            discs.add(x + math.cos(b) * 1.0, y + math.sin(b) * 1.0, 0.5, (weight or 0.95) * fade, leaf["color"])

    def shadows(self, shadow: Discs) -> None:
        for leaf in self.leaves:
            height = self.falling(leaf)
            if height >= 0:
                lift = 1.5 + height * 6
                self.shape(leaf, shadow, lift * SHADOW_DX, lift * SHADOW_DY, 0.2)

    def above(self, frame: np.ndarray, t: float) -> None:
        discs = Discs()
        for leaf in self.leaves:
            height = self.falling(leaf)
            if height < 0:
                continue
            # Fluttering down: swinging side to side as it falls
            swing = math.sin(self.frame_count * 0.25 + leaf["spin"]) * 1.5 * height
            self.shape(leaf, discs, swing, -height * 2)
        discs.paint(frame)


Pond.EVENTS = {
    "feeding": (Feeding, 3),
    "rain": (Rain, 2),
    "dragonfly": (Dragonfly, 2),
    "heron": (Heron, 2),
    "frog": (Frog, 2),
    "leaves": (Leaves, 2),
}

SCENE = Pond


def main() -> None:
    run_scene(Pond, "Koi pond LED matrix display")


if __name__ == "__main__":
    main()
