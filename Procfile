web: gunicorn -w 3 -k uvicorn.workers.UvicornWorker --max-requests 1000 --max-requests-jitter 200 --log-level info --timeout 60 --log-config logging.ini app.main:app
