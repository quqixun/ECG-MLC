FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.1.0-cuda10.0-py3

ADD . /
WORKDIR /

RUN chmod +x ./run.sh
CMD ['sh', 'run.sh']