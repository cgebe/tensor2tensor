FROM python:3-stretch

MAINTAINER Christoph Gebendorfer

RUN pip3 install sacrebleu

COPY . /etc/pscripts
ENV PYTHONPATH /etc/tensor2tensor
ENV TMP_DIR /tmp/t2t_datagen

RUN mkdir -p /etc/tensor2tensor/data
RUN mkdir -p /etc/tensor2tensor/train
RUN mkdir -p /etc/tensor2tensor/decode
ENV DATA_DIR /etc/tensor2tensor/data
ENV TRAIN_DIR /etc/tensor2tensor/train
ENV DECODE_DIR /etc/tensor2tensor/decode
VOLUME ["/etc/tensor2tensor/data", "/etc/tensor2tensor/train", "/etc/tensor2tensor/decode"]

WORKDIR /etc/pscripts

ENTRYPOINT ["/bin/bash"]
