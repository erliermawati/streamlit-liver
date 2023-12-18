import numpy as np
from flask import flask, request, render_templat
import pickle

app = flask(_name_)
model = pickle.load(open('model_new.pkl', 'rb'))
 
 @app.route('/')
 def home():
     return render_template('home.html')
 
 @app.route('/about')
 def about():
     return render_template('about.html')
 
 @app.route('/form_predict')
 def form_predict():
     return render_templat('prdict.html')
 
 @app.route('/predict', method=['POST'])
 def predict():
     features = [float(x) for x in request.form.values()]
     final_features = [np.array(features)]
     prediction = model.predict(final_features)
     
     output = round(prediction[0], 2)
     
     if output ==1:
         out = "Anda terkena Liver"
    else:
        out = "Anda tidak terkena Liver"
    
    return render_templat('results_predict.html', prediction_text="{}",format(out))

if_name_ == "_main_":
    app.run(debug=True)