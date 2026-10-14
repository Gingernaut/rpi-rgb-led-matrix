#!/usr/bin/env python
"""Fireworks: rockets launch from the bottom, trail sparks, and burst into
particles that drift down under gravity and fade.

Bursts come in several styles, each with its own shape so they stay distinct
at this resolution: a filled peony, a thin (sometimes tilted) ring, a two-tone
shell with a core, a palm of a few heavy comets, a gold willow that hangs and
drips, and a silver crackle that glitters out. Stars start white-hot, and
every minute or so the show ends in a finale flurry.

Requires the rgbmatrix Python bindings (make build-python).
"""

import math
import random

from matrix_common import fade_lut, mix, put_max, run_scene, scale

FRAME_SECONDS = 2 / 75  # 37.5 fps

TAU = 2 * math.pi

# Per-frame physics, in pixels: gravity on rockets and on burst stars, and the
# fraction of speed stars keep each frame (air drag). A star flung at speed v
# travels at most v / (1 - DRAG) pixels, so a 0.85 star reaches ~12 px out.
ROCKET_GRAVITY = 0.018
STAR_GRAVITY = 0.008
DRAG = 0.93

# Bursts happen this far down the panel (fractions of its height), rockets
# launch this far across it
BURST_Y = (0.22, 0.45)
LAUNCH_X = (0.15, 0.85)
# Seconds between launches, and every so often a finale flurry: a burst of
# launches (sometimes two rockets at once), then a pause
LAUNCH_GAP = (0.5, 1.6)
FINALE_EVERY = (45.0, 75.0)
FINALE_LAUNCHES = (10, 14)
FINALE_GAP = (0.1, 0.3)
FINALE_PAIR_CHANCE = 0.35
FINALE_PAUSE = 4.0

# Burst stars start white and ease to their color over this many frames
HOT_FRAMES = 6
# Crackle stars glitter for this last fraction of their life
GLITTER_FROM = 0.55
# Palm and willow stars stop dropping embers while more particles than this
# are alive, which caps the cost of a finale's busiest frames
EMBER_LIMIT = 700

# Brightness every lit pixel keeps per frame: short glowing trails. Channels
# that fade below TRAIL_CUTOFF go black: very dim LEDs look muddy.
TRAIL_FADE = 0.74
TRAIL_CUTOFF = 8
FADE_LUT = fade_lut(TRAIL_FADE, TRAIL_CUTOFF)

WHITE = (255, 255, 255)
SILVER = (200, 210, 255)
GOLD = (255, 150, 20)
ROCKET = (255, 220, 160)
SPARK = (255, 120, 20)
# Star colors, tuned to look saturated on LEDs. No yellow or orange: gold
# is the willow's, and the rocket sparks are orange.
COLORS = [
    (255, 20, 10),  # red
    (40, 255, 40),  # green
    (20, 210, 255),  # cyan
    (40, 70, 255),  # blue
    (160, 40, 255),  # purple
    (255, 30, 170),  # magenta
]

# The burst's flash: a bright cross with a dimmer diamond around it
FLASH = [((0, 0), 1.0), ((1, 0), 1.0), ((-1, 0), 1.0), ((0, 1), 1.0), ((0, -1), 1.0),
         ((1, 1), 0.5), ((-1, 1), 0.5), ((1, -1), 0.5), ((-1, -1), 0.5),
         ((2, 0), 0.4), ((-2, 0), 0.4), ((0, 2), 0.4), ((0, -2), 0.4)]


class Particle:
    """A burst star, rocket spark, or ember. trail is the chance per frame of
    dropping an ember behind it; hot stars start white."""

    __slots__ = ("x", "y", "vx", "vy", "life", "lifetime", "color", "gravity", "drag",
                 "hot", "glitter", "trail")

    def __init__(self, x: float, y: float, vx: float, vy: float, lifetime: int, color,
                 gravity: float = STAR_GRAVITY, drag: float = DRAG, hot: bool = False,
                 glitter: bool = False, trail: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = lifetime
        self.lifetime = lifetime
        self.color = color
        self.gravity = gravity
        self.drag = drag
        self.hot = hot
        self.glitter = glitter
        self.trail = trail


class Rocket:
    __slots__ = ("x", "y", "vx", "vy", "burst")

    def __init__(self, x: float, y: float, burst_y: float, burst) -> None:
        self.x = x
        self.y = y
        # Launch speed to rise just to burst_y before gravity stops it
        self.vy = -math.sqrt(2 * ROCKET_GRAVITY * (y - burst_y))
        self.vx = random.uniform(-0.08, 0.08)
        self.burst = burst


class Fireworks:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.fb = bytearray(width * height * 3)
        self.rockets: list[Rocket] = []
        self.particles: list[Particle] = []
        self.last_color = COLORS[0]
        self.last_style = None
        self.next_launch = 0
        self.finale_at = self.seconds_to_frames(random.uniform(*FINALE_EVERY))
        self.finale_left = 0
        self.frame_count = 0

    @staticmethod
    def seconds_to_frames(seconds: float) -> int:
        return max(1, round(seconds / FRAME_SECONDS))

    def pick_color(self):
        """A star color, never the same as the last burst's."""
        self.last_color = random.choice([c for c in COLORS if c != self.last_color])
        return self.last_color

    def shell(self, x: float, y: float, count: int, speed: float, lifetime: tuple[int, int],
              color, fill: float = 0.0, **star) -> None:
        """Fling count stars out from (x, y), up to speed. fill 0 keeps them
        all at the shell's edge; toward 1 they also fill its inside."""
        for _ in range(count):
            angle = random.uniform(0, TAU)
            v = speed * (1 - fill * random.random()) * random.uniform(0.93, 1.0)
            self.particles.append(Particle(x, y, math.cos(angle) * v, math.sin(angle) * v,
                                           random.randint(*lifetime), color, hot=True, **star))

    # --- Burst styles: each adds stars around (x, y) ---------------------------

    def peony(self, x: float, y: float) -> None:
        """A big filled sphere of one color."""
        self.shell(x, y, random.randint(70, 90), random.uniform(0.8, 0.95), (40, 60),
                   self.pick_color(), fill=0.75)

    def ring(self, x: float, y: float) -> None:
        """An even circle that expands as one, often tilted into an ellipse."""
        color = self.pick_color()
        count = 40
        speed = random.uniform(0.8, 0.95)
        # Squash it vertically (tilted away from us), then lean it a little
        tilt = random.choice((1.0, random.uniform(0.4, 0.75)))
        lean = random.uniform(-0.5, 0.5)
        cos_lean, sin_lean = math.cos(lean), math.sin(lean)
        for i in range(count):
            angle = TAU * i / count
            u, w = math.cos(angle) * speed, math.sin(angle) * speed * tilt
            vx, vy = u * cos_lean - w * sin_lean, u * sin_lean + w * cos_lean
            self.particles.append(Particle(x, y, vx, vy, random.randint(45, 52), color, hot=True))

    def two_tone(self, x: float, y: float) -> None:
        """A wide shell of one color around a tight core of another."""
        self.shell(x, y, 50, random.uniform(0.85, 0.95), (45, 55), self.pick_color(), fill=0.15)
        self.shell(x, y, 30, random.uniform(0.35, 0.45), (40, 50), self.pick_color(), fill=0.3)

    def palm(self, x: float, y: float) -> None:
        """A few heavy comets that arc out and droop, trailing sparks."""
        color = self.pick_color()
        count = random.randint(6, 9)
        start = random.uniform(0, TAU)
        for i in range(count):
            angle = start + TAU * (i + random.uniform(-0.2, 0.2)) / count
            v = random.uniform(0.9, 1.05)
            self.particles.append(Particle(x, y, math.cos(angle) * v, math.sin(angle) * v - 0.15,
                                           random.randint(50, 60), color, gravity=0.014,
                                           drag=0.95, hot=True, trail=0.6))

    def willow(self, x: float, y: float) -> None:
        """Gold streamers that droop, hang for a long time, and drip embers."""
        self.shell(x, y, 50, random.uniform(0.7, 0.85), (85, 115), GOLD, fill=0.5,
                   gravity=0.011, drag=0.91, trail=0.15)

    def crackle(self, x: float, y: float) -> None:
        """A silver burst that glitters as it dies."""
        self.shell(x, y, 70, random.uniform(0.8, 0.95), (50, 65), SILVER, fill=0.6, glitter=True)

    def launch(self) -> None:
        # Style weights; never the same style twice in a row
        styles = {self.peony: 3, self.ring: 2, self.two_tone: 2,
                  self.palm: 2, self.willow: 1, self.crackle: 2}
        styles.pop(self.last_style, None)
        self.last_style = random.choices(list(styles), weights=list(styles.values()))[0]
        x = random.uniform(self.width * LAUNCH_X[0], self.width * LAUNCH_X[1])
        burst_y = random.uniform(self.height * BURST_Y[0], self.height * BURST_Y[1])
        self.rockets.append(Rocket(x, self.height, burst_y, self.last_style))

    def schedule(self) -> None:
        """Launch rockets: a steady trickle, then a finale flurry and a pause."""
        self.frame_count += 1
        if self.frame_count == self.finale_at:
            self.finale_left = random.randint(*FINALE_LAUNCHES)
            self.finale_at += self.seconds_to_frames(random.uniform(*FINALE_EVERY))
            self.next_launch = self.frame_count  # start it now
        if self.frame_count < self.next_launch:
            return
        self.launch()
        if self.finale_left:
            if random.random() < FINALE_PAIR_CHANCE:
                self.launch()
            self.finale_left -= 1
            gap = random.uniform(*FINALE_GAP) if self.finale_left else FINALE_PAUSE
        else:
            gap = random.uniform(*LAUNCH_GAP)
        self.next_launch = self.frame_count + self.seconds_to_frames(gap)

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.schedule()
        fb = self.fb = self.fb.translate(FADE_LUT)
        particles = self.particles

        rising = []
        for rocket in self.rockets:
            rocket.x += rocket.vx
            rocket.y += rocket.vy
            rocket.vy += ROCKET_GRAVITY
            if rocket.vy < 0:
                rising.append(rocket)
                put_max(fb, width, height, int(rocket.x), int(rocket.y), ROCKET)
                if random.random() < 0.6:
                    particles.append(Particle(rocket.x + random.uniform(-0.4, 0.4), rocket.y + 1,
                                              random.uniform(-0.05, 0.05), random.uniform(0.0, 0.15),
                                              random.randint(6, 12), SPARK))
                continue
            # At the top of its climb: burst, with a brief flash
            rocket.burst(rocket.x, rocket.y)
            x, y = int(rocket.x), int(rocket.y)
            for (dx, dy), brightness in FLASH:
                put_max(fb, width, height, x + dx, y + dy, scale(WHITE, brightness))
        self.rockets = rising

        alive = []
        embers = len(particles) < EMBER_LIMIT
        for p in particles:
            p.life -= 1
            if p.life <= 0:
                continue
            p.vx *= p.drag
            p.vy = p.vy * p.drag + p.gravity
            p.x += p.vx
            p.y += p.vy
            if p.y >= height:
                continue
            alive.append(p)
            if p.trail and embers and random.random() < p.trail:
                # An ember that hangs where the star was, then sinks and fades
                alive.append(Particle(p.x, p.y, 0.0, 0.02, random.randint(8, 16),
                                      scale(p.color, 0.6), gravity=0.003, drag=0.9))
            # Compare the floats: int() would round -0.5 onto the edge
            if not (0 <= p.x < width and p.y >= 0):
                continue
            remaining = p.life / p.lifetime
            age = p.lifetime - p.life
            if p.hot and age < HOT_FRAMES:
                color, brightness = mix(WHITE, p.color, age / HOT_FRAMES), 1.0
            elif p.glitter and remaining < GLITTER_FROM:
                roll = random.random()
                if roll < 0.2:
                    color, brightness = WHITE, 1.0  # a pop
                elif roll < 0.6:
                    continue  # twinkling off this frame
                else:
                    color, brightness = p.color, 0.45
            else:
                color, brightness = p.color, remaining ** 0.6
            # put_max, inlined: in a finale this runs for ~1000 stars a frame
            i = (int(p.y) * width + int(p.x)) * 3
            r, g, b = color
            r, g, b = int(r * brightness), int(g * brightness), int(b * brightness)
            if r > fb[i]:
                fb[i] = r
            if g > fb[i + 1]:
                fb[i + 1] = g
            if b > fb[i + 2]:
                fb[i + 2] = b
        self.particles = alive

        return fb


SCENE = Fireworks


def main() -> None:
    run_scene(Fireworks, "Fireworks LED matrix display")


if __name__ == "__main__":
    main()
