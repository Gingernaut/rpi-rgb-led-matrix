import uvicorn

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        "create_app:app",
        host="0.0.0.0",
        port=8000,
        debug=True,
        access_log=True,
        workers=1,
    )
