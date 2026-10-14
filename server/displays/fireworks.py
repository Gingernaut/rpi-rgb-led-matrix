#!/usr/bin/env python
"""Fireworks: rockets launch from the bottom, trail sparks, and burst into
particles that drift down under gravity and fade.

Bursts come in several styles (peony, ring, two-tone, gold willow, and a
crackle that glitters out), with a flurry finale every minute or so.

Requires the rgbmatrix Python bindings (make build-python).
"""

import math
import random

from matrix_common import fade_lut, hsv, mix, put_max, run_scene, scale

FRAME_SECONDS = 2 / 75  # 37.5 fps

# Per-frame physics, in pixels: gravity on rockets and on burst particles,
# and the fraction of speed burst particles keep each frame (air drag)
ROCKET_GRAVITY = 0.012
PARTICLE_GRAVITY = 0.006
DRAG = 0.95

# Bursts happen this far below the top edge, in pixels
BURST_Y = (5, 14)
# Seconds between launches, and every so often a finale flurry
LAUNCH_GAP = (0.5, 1.8)
FINALE_EVERY = (45.0, 75.0)
FINALE_ROCKETS = (7, 11)
FINALE_GAP = (0.1, 0.35)

FADE_LUT = fade_lut(0.74)  # short glowing trails behind every particle

WHITE = (255, 255, 255)
GOLD = (255, 160, 30)
SPARK = (255, 120, 20)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "lifetime", "color", "gravity", "drag", "glitter")

    def __init__(self, x, y, vx, vy, lifetime, color, gravity=PARTICLE_GRAVITY, drag=DRAG, glitter=False):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = lifetime
        self.lifetime = lifetime
        self.color = color
        self.gravity = gravity
        self.drag = drag
        self.glitter = glitter


class Rocket:
    __slots__ = ("x", "y", "vx", "vy", "burst")

    def __init__(self, x: float, y: float, burst_y: float, burst) -> None:
        self.x = x
        self.y = y
        # Launch speed to rise just to burst_y before gravity stops it
        self.vy = -math.sqrt(2 * ROCKET_GRAVITY * (y - burst_y))
        self.vx = random.uniform(-0.08, 0.08)
        self.burst = burst


def vivid() -> tuple[int, int, int]:
    return hsv(random.random(), random.uniform(0.75, 1.0))


class Fireworks:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.fb = bytearray(width * height * 3)
        self.rockets: list[Rocket] = []
        self.particles: list[Particle] = []
        self.next_launch = 0
        self.finale_at = self.seconds_to_frames(random.uniform(*FINALE_EVERY))
        self.finale_left = 0
        self.frame_count = 0

    @staticmethod
    def seconds_to_frames(seconds: float) -> int:
        return max(1, round(seconds / FRAME_SECONDS))

    # --- Burst styles: each adds particles around (x, y) -----------------------

    def peony(self, x: float, y: float) -> None:
        """A filled sphere of one color."""
        color = vivid()
        for _ in range(random.randint(40, 60)):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.15, 0.6)
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed,
                                           random.randint(35, 55), color))

    def ring(self, x: float, y: float) -> None:
        """An even circle that expands as one."""
        color = vivid()
        count = 28
        speed = random.uniform(0.45, 0.6)
        for i in range(count):
            angle = 2 * math.pi * i / count
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed, 45, color))

    def two_tone(self, x: float, y: float) -> None:
        """A fast outer shell around a slower core of another color."""
        outer = vivid()
        inner = hsv(random.random(), 0.6)
        for _ in range(55):
            angle = random.uniform(0, 2 * math.pi)
            fast = random.random() < 0.55
            speed = random.uniform(0.45, 0.6) if fast else random.uniform(0.1, 0.3)
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed,
                                           random.randint(40, 55), outer if fast else inner))

    def willow(self, x: float, y: float) -> None:
        """Gold streamers that droop and hang for a long time."""
        for _ in range(45):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.25, 0.55)
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed,
                                           random.randint(80, 110), GOLD, gravity=0.009, drag=0.93))

    def crackle(self, x: float, y: float) -> None:
        """A white-hot burst that glitters as it dies."""
        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.2, 0.6)
            self.particles.append(Particle(x, y, math.cos(angle) * speed, math.sin(angle) * speed,
                                           random.randint(45, 65), mix(WHITE, vivid(), 0.3), glitter=True))

    def launch(self) -> None:
        burst = random.choices(
            [self.peony, self.ring, self.two_tone, self.willow, self.crackle],
            weights=[3, 2, 2, 2, 2],
        )[0]
        x = random.uniform(self.width * 0.15, self.width * 0.85)
        self.rockets.append(Rocket(x, self.height, random.uniform(*BURST_Y), burst))

    def schedule(self) -> None:
        """Launch rockets: a steady trickle, then a finale flurry and a pause."""
        self.frame_count += 1
        if self.frame_count == self.finale_at:
            self.finale_left = random.randint(*FINALE_ROCKETS)
            self.finale_at += self.seconds_to_frames(random.uniform(*FINALE_EVERY))
        if self.frame_count < self.next_launch:
            return
        self.launch()
        if self.finale_left:
            self.finale_left -= 1
            gap = random.uniform(*FINALE_GAP) if self.finale_left else 3.0  # quiet after the finale
        else:
            gap = random.uniform(*LAUNCH_GAP)
        self.next_launch = self.frame_count + self.seconds_to_frames(gap)

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.schedule()
        fb = self.fb = self.fb.translate(FADE_LUT)

        for rocket in list(self.rockets):
            rocket.x += rocket.vx
            rocket.y += rocket.vy
            rocket.vy += ROCKET_GRAVITY
            if rocket.vy >= 0:
                self.rockets.remove(rocket)
                rocket.burst(rocket.x, rocket.y)
                # A brief flash where it burst
                for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    put_max(fb, width, height, int(rocket.x) + dx, int(rocket.y) + dy, WHITE)
                continue
            put_max(fb, width, height, int(rocket.x), int(rocket.y), (255, 220, 160))
            if random.random() < 0.6:
                self.particles.append(Particle(rocket.x + random.uniform(-0.4, 0.4), rocket.y + 1,
                                               random.uniform(-0.05, 0.05), random.uniform(0.0, 0.15),
                                               random.randint(6, 12), SPARK))

        alive = []
        for p in self.particles:
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
            remaining = p.life / p.lifetime
            if p.glitter and remaining < 0.6:
                if random.random() < 0.5:
                    continue  # twinkling off this frame
                brightness = 1.0
            else:
                brightness = remaining ** 0.6
            put_max(fb, width, height, int(p.x), int(p.y), scale(p.color, brightness))
        self.particles = alive

        return fb


SCENE = Fireworks


def main() -> None:
    run_scene(Fireworks, "Fireworks LED matrix display")


if __name__ == "__main__":
    main()
