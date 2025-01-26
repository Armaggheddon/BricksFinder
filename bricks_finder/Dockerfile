FROM python:3.12.3

WORKDIR /bricks_finder

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY . .

CMD ["python", "-u", "src/main.py"]