# Set the Base Image
FROM python:3.12-slim

WORKDIR /app/

COPY requirements-docker.txt .

RUN pip install -r requirements-docker.txt

COPY ./data/external/plot_data.csv ./data/external/plot_data.csv
COPY ./data/processed/test_data.csv ./data/processed/test_data.csv
COPY ./reports/ ./reports/


COPY ./models/ ./models/
COPY ./app.py ./app.py

EXPOSE 5000

CMD ["streamlit","run","app.py", "--server.port","5000","--server.address","127.0.0.1"]
