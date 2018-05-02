FROM debian:jessie

MAINTAINER Christoph Gebendorfer

COPY . /etc/pscripts
ENV PYTHONPATH /etc/tensor2tensor
ENV TMP_DIR /tmp/t2t_datagen

RUN mkdir -p /etc/tensor2tensor/decode
ENV DECODE_DIR /etc/tensor2tensor/decode
VOLUME ["/etc/tensor2tensor/decode"]

WORKDIR /etc/pscripts

ENTRYPOINT ["/bin/bash"]
