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
    # Get symptom inputs from form
    fever = int(request.form.get('fever', 0))
    cough = int(request.form.get('cough', 0))
    headache = int(request.form.get('headache', 0))
    fatigue = int(request.form.get('fatigue', 0))
    sore_throat = int(request.form.get('sore_throat', 0))
    runny_nose = int(request.form.get('runny_nose', 0))
    short_breath = int(request.form.get('short_breath', 0))
    chest_pain = int(request.form.get('chest_pain', 0))
    vomiting = int(request.form.get('vomiting', 0))
    body_ache = int(request.form.get('body_ache', 0))

    # Combine input into one list
    input_data = [[
        fever, cough, headache, fatigue,
        sore_throat, runny_nose, short_breath,
        chest_pain, vomiting, body_ache
    ]]

    # Use model to predict diagnosis
    result = model.predict(input_data)[0]

    # Show prediction on webpage
    return render_template('index.html', prediction=result)

# Run the Flask web app
if __name__ == '__main__':
    app.run(debug=True)