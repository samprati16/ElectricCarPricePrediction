from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('EVcar_model','rb'))

@app.route('/')
def home():
    return render_template('new.html')

@app.route('/predict',methods=['POST'])
def predict():
    lrmodel = [float(x) for x in request.form.values()]
    price = [np.array(lrmodel)]
    result = model.predict(price)
    val = round(result[0], 2)
    return render_template('new.html',output="{}".format(val))

if  __name__ == '__main__':
    app.run(debug=True)