FROM python:3.12-slim

WORKDIR /app

COPY services/fund-intel/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/fund-intel ./services/fund-intel
COPY shared ./shared

ENV PYTHONPATH=/app

EXPOSE 8005

CMD ["gunicorn", "services.fund_intel.app:app", "-b", "0.0.0.0:8005", "-w", "2"]