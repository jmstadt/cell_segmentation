from flask import Flask, request
from fastai.text import *
import requests
import os.path

path = ''
export_file_url = 'https://www.dropbox.com/s/awbay851mj1sgj1/movies_export.pkl?dl=1'
export_file_name = 'movies_export.pkl'

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

learn = load_learner(path, export_file_name)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        review_text = request.form.get('review_text')
        preds = learn.predict(review_text)

        return '''<h1>The review text is: {}</h1><h1>The prediction value is: {}</h1>'''.format(review_text, preds)

    return '''<form method="POST">
                  Review_text: <input type="text" name="review_text"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''


