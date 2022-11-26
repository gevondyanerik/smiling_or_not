from inference import get_predict
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
    imagefile = request.files['imagefile']

    if not imagefile.filename.endswith(('.jpg', '.png', '.jpeg')):
        return render_template('index.html', predict='Input must be jpg/png/jpeg file!')
    
    image = Image.open(imagefile.stream)

    predict = get_predict(image=image)

    return render_template('index.html', predict=predict)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
