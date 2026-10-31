import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        data = {
            'year': int(request.form.get('year', 0)),
            'engine_hp': float(request.form.get('engine_hp', 0)),
            'make': request.form.get('make', ''),
            'transmission': request.form.get('transmission', '')
        }
        
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
