import os
import numpy as np 
from flask import Flask,request,render_template

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app=Flask(__name__)
model=load_model('arrhythmia.h5')

@app.route("/")
def about():
    return render_template("home.html")
@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/types")
def types():
    return render_template("types.html")

@app.route("/info")
def information():
    return render_template("info.html")

@app.route("/predict_base")
def test():
    return render_template("predict_base.html")

@app.route("/predict_base",methods=["GET","POST"])
def upload():
    if request.method == 'POST':
        f=request.files['file'] #requesting the file
        basepath=os.path.dirname('__file__')#storing the file directory
        filepath=os.path.join(basepath,"uploads",f.filename)#storing the file in uploads folder
        f.save(filepath)#saving the file
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to array
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        
        pred=model.predict(x)#predicting classes
        y_pred = np.argmax(pred)
        print("prediction",y_pred)#printing the prediction
    
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction',
       'Premature Ventricular Contractions', 'Right Bundle Branch Block','Ventricular Fibrillation']
        result=str(index[y_pred])

        return result
    return None
#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(debug=True)
    #app.run(host='0.0.0.0', port=8000)