FROM nvcr.io/nvidia/pytorch:24.11-py3

WORKDIR /clip_finetune

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /src ./src

CMD ["python", "-u", "src/entrypoint.py"]