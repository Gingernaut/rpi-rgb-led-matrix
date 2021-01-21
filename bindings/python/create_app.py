from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import os


class FileSelection(BaseModel):
    python_file: str


class ImageScroller(FileSelection):
    python_file: str = "song_scroller2"
    image_path: str
    artist: str
    title: str


def get_script_args_for_mode(mode: str) -> str:
    if isinstance(mode, ImageScroller):
        return (
            " -i "
            + f"./custom_displays/{mode.image_path}"
            + " -a "
            + f"'{mode.artist}'"
            + " -s "
            + f"'{mode.title}'"
        )

    return ""


def create_app():

    app = FastAPI()

    @app.post("/display")
    async def set_display(mode: Union[FileSelection, ImageScroller]):
        base_script = str(
            f"sudo python3 custom_displays/{mode.python_file}.py --led-rows 32 --led-cols 64 --led-gpio-mapping=adafruit-hat --led-show-refresh --led-slowdown-gpio=3 "
        )
        script_args = get_script_args_for_mode(mode)
        script = base_script + " " + script_args
        print("executing")
        print(script)
        os.system(script)
        return {"status": "set mode"}

    return app


# called from uvicorn in `main.py`
if __name__ == "create_app":
    app = create_app()
