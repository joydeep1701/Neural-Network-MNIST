from flask import Flask, redirect, request, jsonify
from flask import render_template, flash
import json
import sys
from tester import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	app.logger.debug("Went to OCR page")
	return render_template('index.html', title='Optical Character Recognition', prediction=None)

@app.route('/_do_ocr', methods=['GET', 'POST'])
def do_ocr():
    """Add two numbers server side, ridiculous but well..."""
    app.logger.debug("Accessed _do_ocr page with image data")
    # flash('Just hit the _add_numbers function')
    # a = json.loads(request.args.get('a', 0, type=str))
    data = request.args.get('imgURI', 0, type=str)
    app.logger.debug("Data looks like " + data)
    index = request.args.get('index', 0, type=int)
    vocab = json.loads(request.args.get('vocab',0,type=str))
    latest_image_array = preprocess(data)
    result = mlp.predict(latest_image_array)

    return jsonify(result="Prediction: "+str(result))
app.run(host='0.0.0.0', port=5000)
#app.run(debug=True)
