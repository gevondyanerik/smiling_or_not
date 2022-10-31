from inference import *
import os
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    
    if not imagefile.filename.endswith('.jpg'):
        return 'Input file must be .jpg format'

    imagefile.save(imagefile.filename)

    result = get_predict(imagefile.filename)

    os.remove(imagefile.filename)

    return result



if __name__ == '__main__':
    app.run(port=777, debug=True)
