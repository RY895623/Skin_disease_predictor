import sys
import os
from dotenv import load_dotenv

# -----------------------
# ENV LOAD
# -----------------------
load_dotenv()

# -----------------------
# PATH FIX (safe imports)
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "app"))

# -----------------------
# IMPORT FASTAPI APP
# -----------------------
from app.main import app

# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )