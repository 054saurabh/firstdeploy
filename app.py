from flask import Flask,render_template,request
import librosa
import numpy as np
import pickle
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'voice_data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model=pickle.load(open('model.pkl','rb'))

def extract_feature1(file_path):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr) 
    scale_mfcc_feature=np.mean(mfcc.T, axis=0)
    
    scale_mfcc_feature = np.array(scale_mfcc_feature.tolist())
    scale_mfcc_feature = scale_mfcc_feature.reshape(1, -1)
    
    return scale_mfcc_feature
    

@app.route('/')
def Home():
    return render_template("index.html")


@app.route('/voice_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        audio_file = request.files['file1']
        if audio_file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(filename)
            
            feature = extract_feature1(filename)
            
            predic = model.predict(feature)[0]
            
            pr_probility=model.predict_proba(feature)
            if predic==0:
                prob=(round(pr_probility[0][0],4))*100
            elif predic==1:
                prob=(round(pr_probility[0][1],4))*100
            return render_template("index.html", predic=predic,prob=prob)
        
        else:
            return render_template("index.html", predic="gdf65")
        
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
