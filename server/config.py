from pydantic import BaseModel


class DisplayConfig(BaseModel):
    rows: int = 32
    cols: int = 64
    gpio_mapping: str = "adafruit-hat"
    brightness: int = 100
    slowdown_gpio: int = 4
    pwm_lsb_nanoseconds: int = 300
    limit_refresh_rate_hz: int = 150

    def to_args(self) -> list[str]:
        """Convert to CLI args for rpi-rgb-led-matrix C binaries (--led-* prefix)."""
        return [
            f"--led-rows={self.rows}",
            f"--led-cols={self.cols}",
            f"--led-gpio-mapping={self.gpio_mapping}",
            f"--led-brightness={self.brightness}",
            f"--led-slowdown-gpio={self.slowdown_gpio}",
            f"--led-pwm-lsb-nanoseconds={self.pwm_lsb_nanoseconds}",
            f"--led-limit-refresh={self.limit_refresh_rate_hz}",
        ]

    def to_python_args(self) -> list[str]:
        """Convert to CLI args for custom Python display scripts."""
        return [
            f"--rows={self.rows}",
            f"--cols={self.cols}",
            f"--gpio-mapping={self.gpio_mapping}",
            f"--brightness={self.brightness}",
            f"--slowdown-gpio={self.slowdown_gpio}",
            f"--pwm-lsb-nanoseconds={self.pwm_lsb_nanoseconds}",
            f"--limit-refresh-rate-hz={self.limit_refresh_rate_hz}",
        ]


config = DisplayConfig()
