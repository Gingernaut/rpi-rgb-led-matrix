import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.config import config
from server.display import DisplayManager


display = DisplayManager(config)

# Animation modes: POST /display/<name> runs server/displays/<script>
ANIMATIONS = {
    "wave1": "wave1.py",
    "plasma": "plasma.py",
    "life": "life.py",
    "planets": "planets.py",
    "matrix-rain": "matrix_rain.py",
    "fireworks": "fireworks.py",
    "reaction-diffusion": "reaction_diffusion.py",
    "aquarium": "aquarium.py",
    "aquarium-night": "aquarium_night.py",
    "iss": "iss.py",
    "flights": "flights.py",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    display.stop()


app = FastAPI(title="Pixel Display", lifespan=lifespan)


class DemoRequest(BaseModel):
    demo: int = 0


@app.post("/display/off")
def display_off():
    display.stop()
    return {"status": "off"}


@app.post("/display/demo")
def display_demo(req: DemoRequest):
    pid = display.start(
        cmd=["./examples-api-use/demo"],
        extra_args=[f"-D{req.demo}"],
    )
    return {"status": "running", "pid": pid, "demo": req.demo}


@app.post("/display/spotify")
def display_spotify():
    if not os.environ.get("CLIENT_ID") or not os.environ.get("CLIENT_SECRET"):
        raise HTTPException(status_code=500, detail="CLIENT_ID and CLIENT_SECRET must be set")

    pid = display.start_python(
        script="server/displays/spotify.py",
        env={
            "CLIENT_ID": os.environ["CLIENT_ID"],
            "CLIENT_SECRET": os.environ["CLIENT_SECRET"],
        },
    )
    return {"status": "running", "pid": pid, "mode": "spotify"}


@app.get("/display/status")
def display_status():
    return {
        "running": display.is_running,
        "pid": display.current_pid,
    }


@app.get("/display/modes")
def display_modes():
    # Every name here can be POSTed to /display/<name>; "off" comes first
    return {"animations": ["off", *ANIMATIONS]}


# Declared last so the fixed /display/... routes above take precedence
@app.post("/display/{mode}")
def display_animation(mode: str):
    script = ANIMATIONS.get(mode)
    if script is None:
        raise HTTPException(status_code=404, detail=f"Unknown mode {mode!r}; see /display/modes")
    pid = display.start_python(script=f"server/displays/{script}")
    return {"status": "running", "pid": pid, "mode": mode}
