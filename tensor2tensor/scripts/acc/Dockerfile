FROM python:3-stretch

MAINTAINER Christoph Gebendorfer

COPY . /etc/scripts

RUN mkdir -p /etc/scripts/data
VOLUME ["/etc/scripts/data" ]

WORKDIR /etc/scripts

ENTRYPOINT ["/bin/bash"]
