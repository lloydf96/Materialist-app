FROM python:3.9-slim-bullseye

RUN mkdir /app

COPY data /app/data
COPY flaskr /app/flaskr
COPY model /app/model
COPY requirements.txt /app/
COPY flask_dummy.py /app/
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR flaskr

EXPOSE 5000:5000

CMD ["python", "flask_app.py","5000"]

