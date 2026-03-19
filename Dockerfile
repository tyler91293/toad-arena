FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV PORT=8765
EXPOSE 8765

CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 30 server:app
