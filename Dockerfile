# This is a potassium-standard dockerfile, compatible with Banana

# Don't change this. Currently we only support this specific base image.
FROM python:3.12-rc-slim-buster

WORKDIR /

RUN apt-get update
RUN apt-get install -y git build-essential cmake

ENV FLASK_APP=voice_of_mogen.api:create_app
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

ADD . .

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

ADD . .

EXPOSE 8080

CMD ["flask", "run"]
