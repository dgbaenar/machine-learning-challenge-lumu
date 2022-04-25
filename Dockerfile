FROM python:3.8

RUN apt update && apt install python-dev -y

COPY requirements.txt /
COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

WORKDIR /app

EXPOSE 8000

RUN chmod +x run.sh
CMD sh run.sh
