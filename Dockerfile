FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app
COPY ./server.py /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python","./server.py"]