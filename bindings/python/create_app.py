from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, List
import multiprocessing
from multiprocessing import Pool
from os import getpid
import time
import subprocess
import psutil


class FileSelection(BaseModel):
    python_file: str


class CustomText(BaseModel):
    python_file: str = "runtext"
    text: str
    size: str = "medium"
    speed: str = "slow"
    color: str = "white"


class ImageScroller(FileSelection):
    python_file: str = "song_scroller2"
    image_path: str
    artist: str
    title: str


class DemoProject(BaseModel):
    demo_title: str


def execute_background_script(script: List[str]):
    print(f"--- {getpid()} executing {script}")
    return subprocess.run(script)


class DisplayThreadManager:
    process = None

    def __init__(self):
        self.process = None

    def display(self, mode) -> str:
        print("Display() is called in process ", getpid())
        self.stop()

        # TODO: clean up any tmp media directory

        script = self.get_script_for_mode(mode)
        self.process = multiprocessing.Process(
            target=execute_background_script, args=(script,)
        )
        self.process.daemon = False
        self.process.start()

        print("started process")

        return " ".join(script)

    def stop(self):
        if self.process and self.process.is_alive():
            print(f"now killing {self.process.pid}")
            self.process.terminate()
            time.sleep(0.1)

        for process in psutil.process_iter():
            if "--led-gpio-mapping=adafruit-hat" in process.cmdline():
                print(f"terminating {' '.join(process.cmdline())}")
                process.terminate()
                time.sleep(0.1)

    def get_script_args_for_mode(
        self, mode: Union[FileSelection, ImageScroller, DemoProject, CustomText]
    ) -> List[str]:
        if isinstance(mode, ImageScroller):
            return [
                "-i",
                f"./custom_displays/{mode.image_path}",
                "-a",
                f"{mode.artist}",
                "-s",
                f"{mode.title}",
            ]

        if isinstance(mode, CustomText):
            return [
                "--text",
                f"{mode.text}",
                "--color",
                f"{mode.color}",
                "--size",
                f"{mode.size}",
                "--speed",
                f"{mode.speed}",
            ]

        if isinstance(mode, DemoProject):
            if mode.demo_title == "conway":
                return ["-D7"]
            if mode.demo_title == "pulsing_colors":
                return ["-D4"]

        return []

    def get_script_for_mode(
        self, mode: Union[FileSelection, ImageScroller, DemoProject, CustomText]
    ) -> List[str]:

        base_script = []
        if isinstance(mode, DemoProject):
            base_script += [
                "sudo",
                "../../examples-api-use/demo",
                "--led-rows",
                "32",
                "--led-cols",
                "64",
                "--led-gpio-mapping=adafruit-hat",
                "--led-slowdown-gpio=3",
            ]
        else:
            base_script += [
                "sudo",
                "/led-matrix/bindings/python/.venv/bin/python3",
                f"custom_displays/{mode.python_file}.py",
                "--led-rows",
                "32",
                "--led-cols",
                "64",
                "--led-gpio-mapping=adafruit-hat",
                "--led-slowdown-gpio=3",
            ]

        return base_script + self.get_script_args_for_mode(mode)


def create_app():

    app = FastAPI()

    pixel_screen = DisplayThreadManager()

    @app.post("/display")
    async def set_display(
        mode: Union[FileSelection, ImageScroller, DemoProject, CustomText]
    ):
        print(f"serving web request inside {getpid()}")
        script = pixel_screen.display(mode)
        return {"status": f"set mode to {mode}", "script": script}

    @app.delete("/display")
    async def turnoff_display():
        print(f"serving web request inside {getpid()}")
        blank_mode = FileSelection(python_file="blank")
        pixel_screen.display(blank_mode)
        return {"status": "turned off display"}

    return app


# called from uvicorn in `main.py`
if __name__ == "create_app":
    app = create_app()
