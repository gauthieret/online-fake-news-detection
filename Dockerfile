FROM python:3.10.6-buster
COPY api /api
COPY ofnd /ofnd
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --reload --host 0.0.0.0 --port $PORT
