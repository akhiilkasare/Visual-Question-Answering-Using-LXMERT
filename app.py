from IPython.display import clear_output, Image, display
import PIL.Image as image
import io
import json
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os
from flask import Flask, request, jsonify, render_template

from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from predict import lxmert
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)

UPLOAD_FOLDER = 'test'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class ClientApp:
    def __init__(self,filename):
        self.filename = filename
        #self.question = 
        self.answer = lxmert(self.filename)

@app.route("/", methods=['GET'])
def home():
    return render_template('index1.html')

@app.route("/predict", methods=['POST'])
def predictRoute():
   
    

    if request.method == 'POST':
        # check if the post request has the file part
        x=request.form['Question']
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
           
    #image = request.json['image']
    clApp=ClientApp(filename)
    result = clApp.answer.prediction(x,filename)

    return result


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






if __name__ == '__main__':
    
    app.run(host='localhost', port=8000, debug=True)
