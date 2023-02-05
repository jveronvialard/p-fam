FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

WORKDIR /app

COPY data/ ./data/

COPY setup.py ./setup.py
COPY requirements.txt ./requirements.txt
COPY pfam/ ./pfam/
COPY tests/ ./tests/
COPY train.py ./train.py
COPY test.py ./test.py
COPY predict.py ./predict.py

RUN conda install pytorch==1.8.1 -c pytorch
RUN pip install -e .