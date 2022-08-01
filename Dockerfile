FROM pytorch/pytorch:latest

ENV TZ=CET-1CEST,M3.5.0,M10.5.0/3
ENV TZ_DJANGO=Europe/Berlin

RUN apt-get update && apt-get install -y \
    python-qt4 \
    python-pyside \
    python-pip \
    python3-pip \
    python3-pyqt5

RUN apt install -y yum-utils
RUN apt-get install -y git
RUN apt-get install -y vim 

WORKDIR /app

RUN pip install --upgrade pip && pip install flask argparse matplotlib torch scikit-learn numpy scipy pandas requests pytorch_lightning pytorch_forecasting requests datetime

COPY . /app

EXPOSE 8888/tcp

RUN echo "Successfully create image!"

CMD python webpage_backend.py


