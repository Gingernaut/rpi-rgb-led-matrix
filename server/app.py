from contextlib import asynccontextmanager

from fastapi import FastAPI
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


@app.get("/display/status")
def display_status():
    return {
        "running": display.is_running,
        "pid": display.current_pid,
    }
