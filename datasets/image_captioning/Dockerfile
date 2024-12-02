FROM nvcr.io/nvidia/pytorch:24.11-py3

WORKDIR /image_captioning

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY src/ .

CMD ["python3", "src/main.py", "--dataset", "armaggheddon97/lego_minifigure_captions"]