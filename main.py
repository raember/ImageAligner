from typing import Set
from flask import Flask, request, send_file, send_from_directory
import cv2
import datetime
import werkzeug.datastructures
import os
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)
from scanner import Transformer

UPLOAD_FOLDER: str = './uploads/'
# EDGED_FOLDER: str = './edged/'
RESULT_FOLDER: str = './result/'
for dir in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.isdir(dir):
        os.mkdir(dir)
ALLOWED_EXTENSIONS: Set[str] = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
optimus_prime = Transformer()


@app.route('/')
def hello_world():
    message = 'Welcome to the classifier'
    return message


@app.route('/rectify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        file: werkzeug.datastructures.FileStorage = request.files['image']
        if file and allowed_file(file.filename):
            date = datetime.datetime.now()
            id = date.strftime("%Y%m%d%H%M%S") + file.filename
            file.save(UPLOAD_FOLDER + id)
            try:
                orig = cv2.imread(UPLOAD_FOLDER + id)
                image, ratio = optimus_prime.create_smaller_copy(orig)
                edged = optimus_prime.detect_edges(image)
                # cv2.imwrite(EDGED_FOLDER + id, edged)
                screenCnt = optimus_prime.find_contours(edged)
                screenCnt = optimus_prime.order_points(screenCnt.reshape(4, 2) * ratio)
                warped = optimus_prime.warp_from_points(orig, screenCnt)
                cv2.imwrite(RESULT_FOLDER + id, warped)
            except Exception as e:
                return "{}".format(e)
            return send_from_directory(RESULT_FOLDER, id)
        else:
            return 'Unsupported filetype'
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=image>
             <input type=submit value=Upload>
        </form>
    '''


# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run()
