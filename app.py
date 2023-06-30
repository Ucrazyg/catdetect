from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename

import tensorflow # Importeer de TensorFlow-bibliotheek
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)

        image_path = "static/uploads/"+filename

        # Laad het voorgeleerde MobileNetV2-model op dat getraind is om katten te detecteren
        model = MobileNetV2(weights='imagenet')

        # Laad en preprocesed de afbeelding
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Classificeert de afbeelding met behulp van het model
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=5)[0]

        # Controleert of een van de hoogste voorspellingen een kat label bevat
        cat_found = any(label == 'tabby' or label == 'tiger_cat' or label == 'Persian_cat' or label == 'Egyptian_cat' or label == 'jaguar' or label == 'leopard' or label == 'lion' or label == 'tiger' for _, label, _ in decoded_preds)

        if cat_found:
            flash('Cat found! 😺')
        else:
            flash('No cat found 😿')

        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)