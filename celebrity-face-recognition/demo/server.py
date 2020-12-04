from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
import numpy as np
import os
import sys
from PIL import Image, ImageDraw

from config.config import Config
from demo.utils.label import read_label
from demo.inference import Inference

cfg = Config()
# PATH_TO_CKPT = os.path.join('pretrained', 'celeb_best_20191220.pt')
PATH_TO_LABELS = os.path.join('pretrained', 'label.txt')
label_dict, NUM_CLASSES = read_label(PATH_TO_LABELS)


INFERENCE = Inference(NUM_CLASSES, cfg, pretrained_model_path='pretrained')


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads/')
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                filename=filename))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]

    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        image = image.convert("RGB")
        boxes, preds, prob = INFERENCE.infer(image)
        # boxes = [np.array([10, 10, 100, 100])]
        pred_with_label = [label_dict[pred] for pred in preds]
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            coord = boxes[i].tolist()
            draw.rectangle(coord, outline=(255, 0, 0), width=6)
            draw.text(coord[:2], f'{pred_with_label[i]}: {prob[i]}')

        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
