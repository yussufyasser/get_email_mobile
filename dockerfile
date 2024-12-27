FROM python:3.8-slim

RUN apt-get update && \
apt-get upgrade -y

RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick
RUN apt-get install -y --fix-missing \    
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev
RUN apt-get install -y --fix-missing pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*


RUN mkdir -p /root/dlib
RUN git clone -b 'v19.24' --single-branch https://github.com/davisking/dlib.git /root/dlib/
RUN cd /root/dlib/ && \
    python3 setup.py install


RUN cd ~

RUN cd ~ && \
    mkdir -p face_recognition && \
    git clone https://github.com/ageitgey/face_recognition.git face_recognition/ && \
    cd face_recognition/ && \
    pip3 install -r requirements.txt && \
    python3 setup.py install



RUN cd /root
COPY . /root
WORKDIR /root

RUN apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx


RUN pip install opencv-python

RUN pip install -r requirements.txt

EXPOSE 8000

CMD [ "python" ,"get_email.py"]