#!/bin/sh
set -e
PORT=${PORT:-8765}
exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 30 server:app
