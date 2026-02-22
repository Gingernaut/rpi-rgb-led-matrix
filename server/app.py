import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.config import config
from server.display import DisplayManager


display = DisplayManager(config)


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
