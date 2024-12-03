FROM nvcr.io/nvidia/pytorch:24.11-py3

WORKDIR /image_captioning

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY src/ .

