#%%
import numpy as np
from flask import Flask, request,render_template
import pickle

#%%
app = Flask(__name__)
model = pickle.load(open('samplehptuning.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
 
    string_lst = eval(request.form.get("variable"))
    input_feature = [float(i) for i in string_lst]
    input_feature = [np.array(input_feature)]
    pred = model.predict(input_feature)

    if pred[0] == 0:
        pred = "False"
    elif pred[0] == 1:
        pred = "True"

    return render_template('index.html', prediction_text='Does this person tend to have heart attack?: {}'.format(pred))

if __name__ == "__main__":
    app.run(debug=True)
