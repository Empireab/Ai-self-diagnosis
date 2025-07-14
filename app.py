from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained AI model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values (0 if not checked)
    symptoms = [
        int(request.form.get('fever', 0)),
        int(request.form.get('cough', 0)),
        int(request.form.get('headache', 0)),
        int(request.form.get('fatigue', 0)),
        int(request.form.get('sore_throat', 0)),
        int(request.form.get('runny_nose', 0)),
        int(request.form.get('short_breath', 0)),
        int(request.form.get('chest_pain', 0)),
        int(request.form.get('vomiting', 0)),
        int(request.form.get('body_ache', 0)),
    ]

    # If no symptom selected, don't show result
    if any(symptoms):
        result = model.predict([symptoms])[0]
        return render_template('index.html', prediction=result)
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
