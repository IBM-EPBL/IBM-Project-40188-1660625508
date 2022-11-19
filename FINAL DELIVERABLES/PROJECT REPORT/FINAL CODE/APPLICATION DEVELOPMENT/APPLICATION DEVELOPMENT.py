import os
import numpy as np
from flask import Flask,request,render_template
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
app=Flask(__name__)
model=load_model('ECG.h5')
@app.route("/")
def about():
    return render_template("about.html")
@app.route("/about")
def home():
    return render_template("about.html")
@app.route("/info")
def information():
    return render_template("info.html")
@app.route("/upload")
def test():
    return render_template("index6.html")
@app.route("/predict", methods=["GET","POST"])
def upload():
    if request.method=="POST":
        f=request.files['file']
        basepath=os.path.dirname('__file__')
        filepath=os.path.join(basepath,"uploads",f.filename)
        f.save(filepath)
        img=load_img(filepath,target_size=(64,64))
        x=img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=model.predict_classes(x)
        print("prediction",pred)
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction','Premature Ventricular Contraction','Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[pred[0]])
        return result
    return None
port=int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(debug=False)
    
