from pydantic import BaseModel


class DisplayConfig(BaseModel):
    rows: int = 32
    cols: int = 64
    gpio_mapping: str = "adafruit-hat"
    brightness: int = 50
    slowdown_gpio: int | None = None

    def to_args(self) -> list[str]:
        """Convert to CLI args understood by rpi-rgb-led-matrix binaries."""
        args = [
            f"--led-rows={self.rows}",
            f"--led-cols={self.cols}",
            f"--led-gpio-mapping={self.gpio_mapping}",
            f"--led-brightness={self.brightness}",
        ]
        if self.slowdown_gpio is not None:
            args.append(f"--led-slowdown-gpio={self.slowdown_gpio}")
        return args


config = DisplayConfig()
