from fastai.vision import *
from fastai.metrics import error_rate
from flask import Flask, request, url_for, flash
from werkzeug import secure_filename
from flask import send_from_directory

import numpy as np
import os
from os import rename, listdir
from PIL import Image

import class_def
from class_def import SegLabelListCustom
from class_def import SegItemListCustom

path = ''
export_file_url = 'https://www.dropbox.com/s/bjszupvu7a15ccb/cell_export.pkl?dl=1'
export_file_name = 'cell_export.pkl'

def down_load_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
            
def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        down_load_file(filename, url)
        return True
    return False

download_if_not_exists(export_file_name, export_file_url)

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)

class SegItemListCustom(SegmentationItemList):
    _label_cls = SegLabelListCustom

learn = load_learner(path, export_file_name)

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = set(['jpg', 'png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
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
            #filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image = open_image(filename)
            image_url = url_for('uploaded_file', filename=filename)
            think = learn.predict(image)
            think_np = np.array(think[1])
            think_np.shape = (256,256)
            think_np = think_np.astype(int)
            think_np[think_np > 0] = 255
            think_im = Image.fromarray((think_np).astype('uint8'), mode='L')
            think_im.save(os.path.join(app.config['UPLOAD_FOLDER'], 'think2_im.png'))
            think_im_url = url_for('uploaded_file', filename='think2_im.png')
            print(think_im_url)
            #image.show(y=learn.predict(image)[0])
            return '''<h1>The cell image is:</h1>
            <img src= "{}" height = "85" width="200"/>
            <h1>The cell segmentation is:</h1>
            <img src= "{}" height = "85" width="200"/>'''.format(image_url, think_im_url)


    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload an image of Cells</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
