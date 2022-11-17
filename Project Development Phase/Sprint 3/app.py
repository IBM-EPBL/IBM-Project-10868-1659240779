# app.py
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import requests
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import sys
sys.path.append('pytorch-image-models')

import timm

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/image')
def classify():
    return render_template('image.html')


@app.route('/preview/<path:path>')
def display_image(path):
    return send_from_directory('static/uploads', path)


def nutrition(fruit):
    response = requests.get("https://www.fruityvice.com/api/fruit/"+fruit)
    return response.json()['nutritions']

@app.route('/predict',methods=['GET','POST'])
def predict():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


    class CustomEfficientNet(nn.Module):
        def __init__(self, model_name= 'tf_efficientnet_b0_ns', pretrained=False):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained=pretrained)
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, 5)

        def forward(self, x):
            x = self.model(x)
            return x

    def get_transforms():
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    class TestDataset(Dataset):
        def __init__(self, transform=None):
            self.transform = transform

        def __getitem__(self, idx):
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image=image)['image']
            return image

    test_dataset = TestDataset(transform = get_transforms())

    model_path = 'best_weight.pth'
    model = CustomEfficientNet('tf_efficientnet_b0_ns', pretrained = False)
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu'))['model'], 
                              strict = True)
    model.eval()
    with torch.no_grad():
        op = model(test_dataset[0].unsqueeze(0))
        op = op.detach().cpu().numpy()
    
    pred = op.argmax(1)
    index = ['apple','banana','orange','pineapple','watermelon']
    fruit = index[pred[0]]
    print(fruit)
    result = nutrition(fruit)
    return render_template("imageprediction.html", nutrition = result, fruit = fruit, filename=filename)


if __name__ == "__main__":
    app.run(debug = True)
