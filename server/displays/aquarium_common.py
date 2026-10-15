"""What the aquarium scenes share: sprites, fish that swim about on their own
or in a school, and a director that starts a random event now and then.

aquarium.py (a bright reef tank) and aquarium_night.py (the deep sea) build
their tanks from these: a tank subclasses Tank, fills self.fish, calls
init_events() and, each frame, direct() and (every few frames) give_room().
"""

import math
import os
import random

from matrix_common import REFRESH_HZ, mix, put

FRAME_SECONDS = 3 / REFRESH_HZ  # every 3rd refresh, ~25 fps

# Fish behavior around events
FLEE_SPEED = 3.5  # times cruising speed, bolting from danger
BAIT_BALL = 0.45  # a frightened school closes up to this much of its spacing
OBSTACLE_LOOK = 14  # how far ahead fish watch for something to steer around

# After EVENT_SECONDS of quiet (random within) one of the tank's EVENTS starts,
# picked by weight and never the same one twice running. One thing happens at
# a time. A tank's RARE_EVENTS keep their own, longer clocks instead, and
# take the next turn once theirs runs out. AQUARIUM_EVENT=<name>
# shows just that one, a couple of seconds in and then after every
# EVENT_TEST_SECONDS of quiet.
EVENT_SECONDS = (20.0, 50.0)
EVENT_TEST_SECONDS = 8.0


# A jellyfish drifting sideways, facing right: its bell ("R" rim, "G" bell,
# "H" core) pulsing through the frames, tentacles ("t") trailing behind
JELLY_FRAMES = [
    [
        "...RRR..",
        "..RGGGR.",
        ".RGGHGGR",
        ".RRRRRRR",
        ".t.t.t.t",
        "t.t.t.t.",
        ".t.t.t..",
        "t.t.t...",
        ".t.t....",
        "t.t.....",
    ],
    [
        "...RRR..",
        "..RGGGR.",
        "..RGHGR.",
        "..RGGGR.",
        "..RRRRR.",
        ".t.t.t..",
        "t.t.t...",
        ".t.t....",
        "t.t.....",
        ".t......",
    ],
    [
        "...RR...",
        "..RGGR..",
        "..RGHR..",
        "..RGGR..",
        "..RGGR..",
        "..RRRR..",
        ".t.t.t..",
        "t.t.t...",
        ".t.t....",
        "t.t.....",
    ],
    [
        "...RRR..",
        "..RGGGR.",
        "..RGHGR.",
        "..RGGGR.",
        "..RRRRR.",
        ".t.t.t..",
        "t.t.t...",
        ".t.t....",
        "t.t.....",
        ".t......",
    ],
]


# An old riveted submarine, facing right (see submarine_frames)
SUB_PALETTE = {
    "Y": (245, 195, 35), "O": (235, 120, 25), "D": (185, 130, 25), "R": (200, 145, 20),
    "P": (255, 240, 170), "S": (160, 160, 150), "L": (255, 255, 220),
}
SUB_HULL = [
    "....................SSS.......",
    "....................S.........",
    "................OOOOSOO.......",
    "................YYYYYYY.......",
    "................YYPYYPY.......",
    "........YYYYYYYYYYYYYYYYYY....",
    ".....YYYYYYYYYYYYYYYYYYYYYYY..",
    "....YYYYPYYYYPYYYYPYYYYPYYYYL.",
    "...OOOOOOOOOOOOOOOOOOOOOOOOOOO",
    "....YRYYRYYRYYRYYRYYRYYRYYRYY.",
    ".....DDDDDDDDDDDDDDDDDDDDDDD..",
    "........DDDDDDDDDDDDDDDDDD....",
]
# The tail, drawn over the hull's first three columns: the rudder, and the
# propeller on its shaft, turning from blades up and down to edge on and back
SUB_PROPS = [
    {6: "S.O", 7: "S.O", 8: "SSO", 9: "S.O", 10: "S.O"},
    {6: "..O", 7: "S.O", 8: "SSO", 9: "S.O", 10: "..O"},
    {6: "..O", 7: "..O", 8: "SSO", 9: "..O", 10: "..O"},
    {6: "..O", 7: "S.O", 8: "SSO", 9: "S.O", 10: "..O"},
]
SUB_LAMP = (28, 7)  # the searchlight, in sprite coordinates


def submarine_frames() -> list[list[str]]:
    """The submarine's frames, the propeller turning through SUB_PROPS."""
    frames = []
    for prop in SUB_PROPS:
        rows = list(SUB_HULL)
        for row, tail in prop.items():
            rows[row] = "".join(t if t != "." else h for t, h in zip(tail, rows[row])) + rows[row][len(tail):]
        frames.append(rows)
    return frames


def frames_for(seconds: float) -> int:
    return max(1, round(seconds / FRAME_SECONDS))


def flip(rows: list[str]) -> list[str]:
    return [row[::-1] for row in rows]


def render_sprite(palette: dict, rows: list[str]) -> list[tuple[int, int, tuple]]:
    """Sprite rows to (dx, dy, color) for the opaque pixels."""
    return [
        (dx, dy, palette[ch])
        for dy, row in enumerate(rows)
        for dx, ch in enumerate(row)
        if ch != "."
    ]


class Sprite:
    """Sprite frames as pixel lists for each facing. Sprites face right; each
    character is a color from the palette, "." is transparent. Given a haze
    color, the colors fade toward it by depth (0 near to 1 far)."""

    def __init__(self, palette: dict, frames: list[list[str]], depth: float = 0.0,
                 haze: tuple | None = None) -> None:
        if haze is not None:
            palette = {ch: mix(color, haze, depth * 0.22) for ch, color in palette.items()}
        # [facing right, facing left][frame]
        self.pixels = [
            [render_sprite(palette, f) for f in frames],
            [render_sprite(palette, flip(f)) for f in frames],
        ]
        self.w = len(frames[0][0])
        self.h = len(frames[0])
        self.frames = len(frames)

    def draw(self, fb: bytearray, width: int, height: int, x: int, y: int, facing: int, frame: int) -> None:
        for dx, dy, color in self.pixels[facing < 0][frame]:
            put(fb, width, height, x + dx, y + dy, color)

    def at(self, x: float, facing: int, point: tuple[int, int]) -> float:
        """The panel x of a point given in the right-facing sprite's coordinates."""
        return x + (point[0] if facing > 0 else self.w - 1 - point[0])


class Fish:
    """A fish cruising back and forth, now and then pausing, turning around or
    changing depth. Given a leader, it keeps its place near that fish instead.
    Events can send it fleeing out of view (flee/calm) or slow it down (pace).

    sprite is anything with w, h and draw() like Sprite's, and two frames
    for the tail beat (more for a slower cycle). cruise is the range its
    cruising speed is picked from. depth is 0 (near) to 1 (far): far fish
    swim a little slower."""

    def __init__(self, sprite, width: int, top: int, bottom: int, depth: float,
                 cruise: tuple[float, float] = (0.12, 0.28),
                 leader: "Fish | None" = None, place: tuple[int, int] = (0, 0)) -> None:
        self.sprite = sprite
        self.w = sprite.w
        self.h = sprite.h
        self.tank_width = width
        self.top = top
        self.bottom = max(top, bottom - self.h)
        self.depth = depth
        self.leader = leader
        self.place = place
        self.offset = place
        # How far the school spreads left and right of this fish, so a leader
        # turns early enough to keep its followers inside the glass, and
        # above and below it, so it steers the whole school clear of things
        self.spread = (0, 0)
        self.stack = (0, 0)
        self.cruise = random.uniform(*cruise) * (1 - 0.3 * depth)
        self.speed = self.cruise
        self.vy = 0.0
        self.beats = getattr(sprite, "frames", 2)
        self.tail = random.uniform(0, 2)  # tail beat phase; the frame is int(tail)
        self.bob = random.uniform(0, 2 * math.pi)
        self.pause = 0
        self.fleeing = False
        self.calm_in = 0  # frames until a fleeing fish heads back
        self.pace = 1.0  # scales its cruising speed
        self.homebody = False  # lives in the tank's scenery, so doesn't steer clear of it
        self.tight = self.tight_to = 1.0  # a leader's school spacing, eased
        if leader:
            self.x = leader.x + place[0]
            self.y = min(max(leader.y + place[1], self.top), self.bottom)
            self.facing = leader.facing
        else:
            self.x = random.uniform(0, width - self.w)
            self.y = random.uniform(self.top, self.bottom)
            self.facing = random.choice((1, -1))
            self.pick_turn()
        self.target_y = self.y

    def pick_turn(self) -> None:
        """Choose where to turn around next: usually just short of the glass,
        sometimes partway across, and sometimes out of view for a while (the
        tank is bigger than the panel)."""
        glass = self.glass(self.facing)
        roll = random.random()
        if roll < 0.15:
            beyond = random.uniform(self.w + 4, self.w + 30)
        elif roll < 0.35:
            beyond = -random.uniform(8, self.tank_width / 2)
        else:
            beyond = random.uniform(-4, 1)
        self.turn_x = glass + beyond * self.facing
        if (self.turn_x - self.x) * self.facing < 8:
            self.turn_x = glass

    def glass(self, facing: int) -> float:
        """Where to turn, swimming facing's way, to keep the school in view."""
        return self.tank_width - self.w - self.spread[1] if facing > 0 else self.spread[0]

    def swim(self) -> None:
        if self.leader:
            self.follow()
        elif self.fleeing:
            self.bolt()
        else:
            self.wander()
        self.x += self.speed * self.facing
        self.y += self.vy
        self.tight += (self.tight_to - self.tight) * 0.05
        # Tail beats faster when swimming faster
        self.tail = (self.tail + 0.08 + self.speed * 0.5) % self.beats

    def glide(self, rate: float = 0.07) -> None:
        """Ease toward target_y, smoothly and at most rate pixels a frame."""
        climb = max(-rate, min(rate, (self.target_y - self.y) * 0.03))
        self.vy += (climb - self.vy) * 0.1

    def wander(self) -> None:
        # Now and then glide to a new depth, slowly and smoothly
        if random.random() < 0.006:
            self.target_y = random.uniform(self.top, self.bottom)
        self.glide()

        # Slow down nearing the turning point, and turn there
        ahead = (self.turn_x - self.x) * self.facing
        want = self.cruise * self.pace * min(1.0, max(0.15, ahead / 8))
        if self.pause:
            # Hover a moment, then maybe head back the other way, if there's
            # room that way
            self.pause -= 1
            want = 0.0
            room = (self.x - self.glass(-self.facing)) * self.facing
            if not self.pause and room >= 8 and random.random() < 0.5:
                self.turn()
        elif ahead <= 0:
            self.turn()
        elif random.random() < 0.002:
            self.pause = random.randint(25, 100)
        self.speed += (want - self.speed) * 0.05

    def turn(self) -> None:
        self.facing = -self.facing
        self.pick_turn()

    def flee(self, from_x: float) -> None:
        """Bolt away from from_x to somewhere out of view, and wait there
        until calm(). A school balls up tight as it goes."""
        self.fleeing = True
        self.calm_in = 0
        self.pause = 0
        self.facing = 1 if self.x + self.w / 2 >= from_x else -1
        # Far enough past the glass for the whole school to be hidden
        if self.facing > 0:
            self.turn_x = self.tank_width + self.spread[0] + random.uniform(3, 15)
        else:
            self.turn_x = -self.w - self.spread[1] - random.uniform(3, 15)
        self.target_y = random.uniform(self.top, self.bottom)
        self.tight_to = BAIT_BALL

    def calm(self, delay: int) -> None:
        """Head back in after delay frames."""
        if self.fleeing:
            self.calm_in = max(1, delay)

    def bolt(self) -> None:
        self.glide(0.12)
        ahead = (self.turn_x - self.x) * self.facing
        want = self.cruise * FLEE_SPEED if ahead > 0 else 0.0
        self.speed += (want - self.speed) * 0.15
        if self.calm_in:
            self.calm_in -= 1
            if not self.calm_in:
                self.fleeing = False
                self.tight_to = 1.0
                # Back toward the middle of the tank from wherever it hid
                self.facing = 1 if self.x + self.w / 2 < self.tank_width / 2 else -1
                self.pick_turn()

    def follow(self) -> None:
        lead = self.leader
        # Turn a moment after the leader does, each in its own time
        if self.facing != lead.facing and random.random() < 0.12:
            self.facing = lead.facing
        # Drift around within the place, so the school loosens and tightens
        if random.random() < 0.004:
            self.offset = (self.place[0] + random.uniform(-1, 1), self.place[1] + random.uniform(-0.7, 0.7))
        # Speed up or ease off to hold the place (never backing up)
        behind = (lead.x + self.offset[0] * lead.tight - self.x) * self.facing
        top_speed = max(lead.cruise * 1.8, lead.speed * 1.3)
        want = max(0.0, min(top_speed, lead.speed + behind * 0.04))
        self.speed += (want - self.speed) * 0.2
        target_y = min(max(lead.y + self.offset[1] * lead.tight, self.top), self.bottom)
        self.vy = max(-0.15, min(0.15, (target_y - self.y) * 0.05))

    def pose(self, t: float) -> tuple[int, int]:
        """Where to draw it: x, and y bobbing gently."""
        return round(self.x), round(self.y + math.sin(t * 1.3 + self.bob) * 0.6)

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        x, y = self.pose(t)
        self.sprite.draw(fb, width, height, x, y, self.facing, int(self.tail))


def school(make_fish, places: list[tuple[int, int]]) -> list[Fish]:
    """A school: a leader from make_fish(None, (0, 0)), kept far enough from
    the glass, surface and floor for the whole school to fit, then a follower
    from make_fish(leader, place) at each place relative to it. The places
    are fixed in the tank, not relative to the leader's heading, so when the
    school turns it wheels around in place and the back becomes the front."""
    leader = make_fish(None, (0, 0))
    leader.spread = (max(-px for px, py in places), max(px for px, py in places))
    leader.stack = (max(-py for px, py in places), max(py for px, py in places))
    leader.top += leader.stack[0]
    leader.bottom -= leader.stack[1]
    leader.y = leader.target_y = min(max(leader.y, leader.top), leader.bottom)
    leader.pick_turn()
    return [leader] + [make_fish(leader, place) for place in places]


# --- Events ------------------------------------------------------------------

BACK = "back"  # drawn behind the scenery and fish
AMONG = "among"  # drawn among the fish, at its depth


class Event:
    """Something passing through the tank for a while. Each frame the tank
    calls step(), then draw() at the event's layer; among the fish, it's drawn
    in front of those deeper than its depth (0 near to 1 far). Once it's over,
    or when another event cuts it short, the tank calls end()."""

    layer = AMONG
    depth = 0.5
    shade = 1.0  # how much light reaches the tank, when something blocks it

    def __init__(self, tank: "Tank") -> None:
        self.tank = tank

    def step(self) -> bool:
        """Advance a frame; False once it's over."""
        return False

    def end(self) -> None:
        """Let the fish go back to their own business."""

    def draw(self, fb: bytearray, t: float) -> None:
        pass

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        """(left, top, right, bottom) boxes the fish steer clear of."""
        return []

    def free_fish(self) -> list[Fish]:
        """The fish that go their own way (school leaders, not followers)."""
        return [fish for fish in self.tank.fish if not fish.leader]


class Tank:
    """The parts of a tank its events and the director rely on. Subclasses
    set EVENTS ({name: (event class, weight when picking one at random)}),
    optionally RARE_EVENTS ({name: (event class, (fewest, most) minutes
    between visits)}) and scenery ((left, top, right, bottom) boxes the fish
    steer clear of, but for homebodies), and width, height, floor (the top
    of the floor) and fish."""

    FRAME_SECONDS = FRAME_SECONDS
    EVENTS: dict = {}
    RARE_EVENTS: dict = {}
    scenery: list[tuple[float, float, float, float]] = []

    width: int
    height: int
    floor: int
    fish: list[Fish]

    def init_events(self) -> None:
        # The event under way (if any), and the quiet frames left before the
        # next one starts
        self.event: Event | None = None
        self.last_event = None
        self.only = os.environ.get("AQUARIUM_EVENT")
        if self.only and self.only not in self.EVENTS and self.only not in self.RARE_EVENTS:
            raise SystemExit(f"AQUARIUM_EVENT must be one of {', '.join([*self.EVENTS, *self.RARE_EVENTS])}")
        self.event_wait = frames_for(2.0 if self.only else random.uniform(*EVENT_SECONDS))
        # Frames until each rare event is due
        self.rare_wait = {name: self.rare_interval(name) for name in self.RARE_EVENTS}

    def rare_interval(self, name: str) -> int:
        return frames_for(60 * random.uniform(*self.RARE_EVENTS[name][1]))

    def start(self, name: str) -> None:
        """Begin the named event (from EVENTS or RARE_EVENTS) now, cutting
        short any under way."""
        if self.event:
            self.event.end()
        self.event = (self.EVENTS.get(name) or self.RARE_EVENTS[name])[0](self)

    def direct(self) -> None:
        """Run the event under way, or count down the quiet to the next."""
        if not self.only:
            for name in self.rare_wait:
                self.rare_wait[name] -= 1
        if self.event:
            if not self.event.step():
                self.event.end()
                self.event = None
            return
        if self.only:
            self.event_wait -= 1
            if self.event_wait <= 0:
                self.event_wait = frames_for(EVENT_TEST_SECONDS)
                self.start(self.only)
            return
        self.event_wait -= 1
        if self.event_wait <= 0:
            self.event_wait = frames_for(random.uniform(*EVENT_SECONDS))
            due = [name for name, wait in self.rare_wait.items() if wait <= 0]
            if due:
                # A rare one's time has come: it takes this turn
                self.rare_wait[due[0]] = self.rare_interval(due[0])
                self.last_event = due[0]
            else:
                names = [name for name in self.EVENTS if name != self.last_event]
                self.last_event = random.choices(names, [self.EVENTS[name][1] for name in names])[0]
            self.start(self.last_event)

    def give_room(self) -> None:
        """Fish that come too close drift apart vertically, so they don't
        swim along stuck together. A school moves aside as one. They also
        steer above or below anything an event asks them to avoid, and the
        scenery."""
        fish = self.fish
        for i, a in enumerate(fish):
            for b in fish[i + 1:]:
                if (a.leader or a) is (b.leader or b):
                    continue  # the same school
                # Compare centers, doubled to stay in whole sprite sizes
                if (abs(2 * (a.x - b.x) + a.w - b.w) < a.w + b.w + 4
                        and abs(2 * (a.y - b.y) + a.h - b.h) < a.h + b.h + 2):
                    up = -1 if 2 * a.y + a.h < 2 * b.y + b.h else 1  # a is above b
                    for f, away in ((a.leader or a, up), (b.leader or b, -up)):
                        f.target_y = min(max(f.y + 3 * away, f.top), f.bottom)
        boxes = [(box, False) for box in self.scenery]
        if self.event:
            boxes += [(box, True) for box in self.event.obstacles()]
        for (left, top, right, bottom), passing in boxes:
            for f in fish:
                if f.leader or (f.homebody and not passing):
                    continue
                # The fish and its school, and the water just ahead of them,
                # so they start to steer in time
                near = f.x - f.spread[0] - (3 if f.facing > 0 else OBSTACLE_LOOK)
                far = f.x + f.w + f.spread[1] + (OBSTACLE_LOOK if f.facing > 0 else 3)
                if (near < right and far > left
                        and f.y - f.stack[0] < bottom + 1 and f.y + f.h + f.stack[1] > top - 1):
                    # Whichever way round is shorter, if there's room
                    above, below = top - 2 - f.h - f.stack[1], bottom + 2 + f.stack[0]
                    go = above if f.y + f.h / 2 < (top + bottom) / 2 and above >= f.top else below
                    f.target_y = min(max(go, f.top), f.bottom)
