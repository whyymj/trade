FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8006

EXPOSE 8006

CMD ["python", "services/llm/app.py"]