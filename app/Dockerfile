FROM python:3.9.12

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]
