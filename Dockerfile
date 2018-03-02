FROM tensorflow/tensorflow:1.6.0

MAINTAINER Christoph Gebendorfer

COPY . /etc/tensor2tensor
ENV PYTHONPATH /etc/tensor2tensor
ENV TMPDIR /tmp/t2t_datagen

RUN mkdir -p /etc/tensor2tensor/data
RUN mkdir -p /etc/tensor2tensor/train
ENV DATADIR /etc/tensor2tensor/data
ENV TRAINDIR /etc/tensor2tensor/train
VOLUME ["/etc/tensor2tensor/data", "/etc/tensor2tensor/train"]

ENTRYPOINT ["/bin/bash"]

t2t-datagen --data_dir=/home/chris/t2t/data --tmp_dir=/tmp/t2t_datagen --problem=translate_deen_legal8k
