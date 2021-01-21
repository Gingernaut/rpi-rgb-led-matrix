from fastapi import FastAPI
from pydantic import BaseModel
import os


class ModeSelection(BaseModel):
    python_file: str


def get_script_args_for_mode(mode: str) -> str:
    if mode == "song_scroller2":
        return (
            " -i "
            + "./custom_displays/bigwild.jpg"
            + " -a "
            + f"'Big Wild'"
            + " -s "
            + f"'when I get there'"
        )




def create_app():

    app = FastAPI()

    @app.post("/display")
    async def set_display(mode: ModeSelection):
        base_script = str(
            f"sudo python3 custom_displays/{mode.python_file}.py --led-rows 32 --led-cols 64 --led-gpio-mapping=adafruit-hat --led-show-refresh --led-slowdown-gpio=3 "
        )
        script_args = get_script_args_for_mode(mode.python_file)
        script = base_script + " " + script_args
        print("executing")
        print(script)
        os.system(script)
        return {"status": "set mode"}

    return app


# called from uvicorn in `main.py`
if __name__ == "create_app":
    app = create_app()
