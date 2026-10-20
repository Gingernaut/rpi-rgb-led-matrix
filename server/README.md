# Display server

`server/app.py` is a FastAPI app that runs one display at a time on the panel, each as a child process; the displays live in `server/displays/` and are registered in `ANIMATIONS`. The server starts with the display off. `run_server.sh` starts it as root for GPIO with `uv run --no-sync` and `PYTHONDONTWRITEBYTECODE=1`, so root never writes into `.venv` or the uv cache.

## Endpoints (port 8000)

| Endpoint | Does |
|---|---|
| `GET /display/status` | Whether a display is running, and its PID |
| `GET /display/modes` | `off`, then the display names |
| `POST /display/<name>` | Start that display, stopping the current one |
| `POST /display/off` | Stop |
| `POST /display/demo` | Run `examples-api-use/demo`; body `{"demo": N}` picks which |
| `POST /display/spotify` | The Spotify display; needs `CLIENT_ID` and `CLIENT_SECRET` in the server's environment |

```bash
curl -s http://<pi>:8000/display/modes
curl -s -X POST http://<pi>:8000/display/aquarium
```

## Developing

- Run `uv` as your user, never with `sudo`: root-owned files break `uv sync`. Use `uv sync --locked`; a plain `uv sync` relocks `uv.lock`, and `git pull` then refuses.
- Preview or time a display without a panel: `uv run server/displays/preview.py <name> --out x.gif` (`--seconds`, `--seed`, `--bench FRAMES`). Each display's docstring lists its environment variables (`AQUARIUM_EVENT`, `TIDE_FEET`, `ISS_START`, …).
- After merging upstream, restore `bindings/python/setup.py`, `bindings/python/Makefile` and the root `Makefile` targets: upstream dropped `make build-python` for scikit-build-core, and the fork keeps it.
- On the Pi this runs as `led-matrix.service`; building, deploying (`hosts/matrix/setup.sh --update`) and the unit are in my dotfiles repo, `setup/infrastructure/RUNBOOK.md`, LED matrix.
