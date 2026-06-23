FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

RUN apt-get update \
	&& apt-get install -y python3 python3-pip

ADD requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt --break-system-packages \
	&& rm /tmp/requirements.txt \
	&& pip3 cache purge \
	&& mkdir /app

WORKDIR /app
ENV HF_HUB_OFFLINE=1
ADD weights /app/weights

ADD *.py /app/

