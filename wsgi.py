from app import app, init_db, init_mediapipe_review_runtime


# Gunicorn loads `application` directly and will not call app.main().
# Run the lightweight runtime initialization explicitly at import time.
init_db()
init_mediapipe_review_runtime()

application = app
