FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PORT=8765
EXPOSE 8765

CMD ["/app/entrypoint.sh"]
