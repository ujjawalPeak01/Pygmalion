FROM nvcr.io/nvidia/tritonserver:23.06-py3

RUN apt update && apt -y install libssl-dev tesseract-ocr libtesseract-dev ffmpeg

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "app.py"]