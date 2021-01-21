import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "create_app:app",
        host="0.0.0.0",
        port=8000,
        debug=True,
        access_log=True,  # disabled in favor of logging middleware,
        workers=1,
    )
