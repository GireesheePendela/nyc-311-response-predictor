FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY best_model.pkl .
COPY feature_cols.pkl .
COPY app.py .
COPY step7_dashboard.html .

EXPOSE 7860

CMD ["python", "app.py"]
