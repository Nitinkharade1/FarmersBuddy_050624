from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
#nitin
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request,session,redirect,flash,url_for
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from datetime import datetime
import json
import os
import pandas as pd
from werkzeug.utils import secure_filename

# import count_vect

from flask import Flask, jsonify, request
import numpy as np

import pandas as pd
import numpy as np



import re



with open('config.json', 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__,template_folder='templates')
app.secret_key = 'super-secret-key'

# Model saved with Keras model.save()
MODEL_PATH ='sani_model20.h5'

# # Load your trained model
# model = load_model(MODEL_PATH)

#####################################################################################################
from tensorflow.keras.models import load_model

model1 = load_model(MODEL_PATH, compile=False)

model1.compile(optimizer='adam',  # Choose an appropriate optimizer
              loss='categorical_crossentropy',  # Use the loss function as per your model's training
              metrics=['accuracy'])  # Include any metrics your model needs
# Assuming TensorFlow Addons is installed
import tensorflow_addons as tfa

optimizer = tfa.optimizers.AdamW(weight_decay=1e-4)  # Adjust weight_decay as needed
model1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



###################################################################################################

# def model_predict(img_path, model1):
#     print(img_path)
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     ## Scaling
#     x = x / 255
#     x = np.expand_dims(x, axis=0)

#     preds = model1.predict(x)
#     preds = np.argmax(preds, axis=1)

#     # Convert predicted class index to string representing class label
#     if preds == 0:
#         preds = "0"
#     elif preds == 1:
#         preds = "1"
#     elif preds == 2:
#         preds = "2"
#     elif preds == 3:
#         preds = "3"
#     elif preds == 4:
#         preds = "4"
#     elif preds == 5:
#         preds = "5"
#     elif preds == 6:
#         preds = "6"
#     elif preds == 7:
#         preds = "7"
#     elif preds == 8:
#         preds = "8"
#     elif preds == 9:
#         preds = "9"
        
#     return preds



def model_predict(img_path, model1):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model1.predict(x)
    preds = np.argmax(preds, axis=1)

    # Convert predicted class index to string representing class label
    class_labels = [
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_healthy",
        "tomato_late_blight",
        "tomato_leaf_mold",
        "tomato_septoria_leaf_spot",
        "tomato_spider_mites",
        "tomato_target_spot",
        "tomato_tomato_mosaic_virus",
        "tomato_tomato_yellow_leaf_curl_virus"
    ]
    preds = class_labels[preds[0]]  # Get the corresponding text label using the predicted index

    return preds


app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = params['gmail_user']
app.config['MAIL_PASSWORD'] = params['gmail_password']
mail = Mail(app)

if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)

class Register(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    uname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    password = db.Column(db.String(10), nullable=False)
    cpassword = db.Column(db.String(10), nullable=False)

class Contact(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    name=db.Column(db.String(50),nullable=False)
    email=db.Column(db.String(50),nullable=False)
    subject=db.Column(db.String(50),nullable=False)
    message=db.Column(db.String(250),nullable=False)

@app.route("/")
def Home():
    return render_template('index.html',params=params)

@app.route("/about")
def About():
    return render_template('about.html',params=params)

@app.route("/album")
def Album():
    return render_template('album.html',params=params)

@app.route("/register", methods=['GET','POST'])
def register():
    if(request.method=='POST'):
        name = request.form.get('name')
        uname = request.form.get('uname')
        email= request.form.get('email')
        password= request.form.get('password')
        cpassword= request.form.get('cpassword')

        user=Register.query.filter_by(email=email).first()
        if user:
            flash('Account already exist!Please login','success')
            return redirect(url_for('register'))
        if not(len(name)) >3:
            flash('length of name is invalid','error')
            return redirect(url_for('register')) 
        if (len(password))<8:
            flash('length of password should be greater than 7','error')
            return redirect(url_for('register'))
        else:
             flash('You have registtered succesfully','success')
            
        entry = Register(name=name,uname=uname,email=email,password=password,cpassword=cpassword)
        db.session.add(entry)
        db.session.commit()
    return render_template('register.html',params=params)

@app.route("/login",methods=['GET','POST'])
def login():
    if (request.method== "GET"):
        if('email' in session and session['email']):
            return render_template('options.html',params=params)
        else:
            return render_template("login.html", params=params)

    if (request.method== "POST"):
        email = request.form["email"]
        password = request.form["password"]
        
        login = Register.query.filter_by(email=email, password=password).first()
        if login is not None:
            session['email']=email
            return render_template('options.html',params=params)
        else:
            flash("plz enter right password or email",'error')
            return render_template('login.html',params=params)

@app.route("/contact",  methods=['GET','POST'])
def contact():
    if(request.method =='POST'):
        name=request.form.get('name')
        email=request.form.get('email')
        subject=request.form.get('subject')
        message=request.form.get('message')
        entry=Contact(name=name,email=email,subject=subject,message=message)
        db.session.add(entry)
        db.session.commit()
    return render_template('contact.html',params=params)


@app.route("/logout", methods = ['GET','POST'])
def logout():
    session.pop('email')
    return redirect(url_for('Home')) 


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", params=params)

@app.route("/options")
def options():
    return render_template("options.html", params=params)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model1)
        result=preds
        return result
    
    
    return None




#######################################################################################################################

from flask import Flask, request, render_template,session,flash,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
import sklearn
import pickle
import pandas as pd
import os
import json
from flask_mail import Mail

# Loading our model:
model = pickle.load(open("sani2.pkl", "rb"))

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predictt", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        
        # Nitrogen
        nitrogen = float(request.form["nitrogen"])
        
        # Phosphorus
        phosphorus = float(request.form["phosphorus"])
        
        # Potassium
        potassium = float(request.form["potassium"])
        
        # Temperature
        temperature = float(request.form["temperature"])
        
        # Humidity Level
        humidity = float(request.form["humidity"])
        
        # PH level
        phLevel = float(request.form["ph-level"])
        
        # Rainfall
        rainfall = float(request.form["rainfall"])
        
        # Making predictions from the values:
        predictions = model.predict([[nitrogen, phosphorus, potassium, temperature, humidity, phLevel, rainfall]])
        
        output = predictions[0]
        finalOutput = output.capitalize()
        
        if (output == "rice" 
            or output == "blackgram" 
            or output == "pomegranate" 
            or output == "papaya"
            or output == "cotton" 
            or output == "orange" 
            or output == "coffee" 
            or output == "chickpea"
            or output == "mothbeans" 
            or output == "pigeonpeas"
            or output == "jute" 
            or output == "mungbeans"
            or output == "lentil" 
            or output == "maize" 
            or output == "apple"):
            cropStatement = finalOutput + " should be harvested. It's a Kharif crop, so it must be sown at the beginning of the rainy season e.g between April and May."
                            

        elif (output == "muskmelon" 
            or output == "kidneybeans" 
            or output == "coconut" 
            or output == "grapes" 
            or output == "banana"):
            cropStatement = finalOutput + " should be harvested. It's a Rabi crop, so it must be sown at the end of monsoon and beginning of winter season e.g between September and October."
            
        elif (output == "watermelon"):
            cropStatement = finalOutput + " should be harvested. It's a Zaid Crop, so it must be sown between the Kharif and rabi season i.e between March and June."
        
        elif (output == "mango"):
            cropStatement = finalOutput + " should be harvested. It's a cash crop and also perennial. So you can grow it anytime."
        
              
                
    return render_template('CropResult.html', prediction_text=cropStatement)

@app.route("/back")
def back():
    return render_template("CropResult.html",params=params)




######################################################################################################################



if __name__ == '__main__':
    app.run(debug=True)

