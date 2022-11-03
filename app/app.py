from inference import *
import os
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
    imagefile = request.files['imagefile']

    if not imagefile.filename.endswith(('.jpg', '.png', '.jpeg')):
        return render_template('index.html', predict='Input must be jpg/png/jpeg file!')

    imagefile.save(imagefile.filename)

    predict = get_predict(imagefile.filename)

    os.remove(imagefile.filename)

    return render_template('index.html', predict=predict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
