from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploadimages'

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
model = tf.keras.models.load_model("d:\Plant Doctor\models\plant_disease_recog_model_pwp.keras")

# Disease labels
labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load disease information
with open("d:\Plant Doctor\plant_disease.json", 'r') as f:
    plant_diseases = json.load(f)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def preprocess_image(image_path):
    """Load and preprocess image for model prediction"""
    img = Image.open(image_path).resize((160, 160))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_predict(image_path):
    """Make prediction and return disease information"""
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Make prediction
        preds = model.predict(img_array)
        pred_index = np.argmax(preds)
        predicted_label = labels[pred_index]
        
        # Find matching disease info
        disease_info = next(
            (d for d in plant_diseases if d['name'] == predicted_label),
            {'name': predicted_label, 'cause': 'Unknown cause', 'cure': 'Consult an expert'}
        )
        
        return disease_info
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {'name': 'Error', 'cause': 'Could not process image', 'cure': 'Try another image'}

@app.route('/upload/', methods=['POST'])
def upload_image():
    if 'img' not in request.files:
        return redirect('/')
        
    file = request.files['img']
    if file.filename == '':
        return redirect('/')
    
    try:
        # Save uploaded file
        filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Get prediction
        prediction = model_predict(save_path)
        
        return render_template(
            'home.html',
            result=True,
            imagepath=f'/uploadimages/{filename}',
            prediction=prediction
        )
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)