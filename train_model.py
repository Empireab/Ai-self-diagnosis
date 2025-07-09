import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

# Step 1: Create expanded dataset
data = {
    'fever':        [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    'cough':        [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    'headache':     [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    'fatigue':      [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    'sore_throat':  [1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    'runny_nose':   [1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
    'short_breath': [0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    'chest_pain':   [0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
    'vomiting':     [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    'body_ache':    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
  'diagnosis': [
    'cold', 'cold', 'Flu',  'Flu', 'Malaria', 'Typhoid', 'Pneumonia', 'Typhoid', 'Flu', 'pneumonia'
  ]
}

df = pd.DataFrame(data)
df.to_csv("diagnosis_data.csv", index=False)

# Step 2: Train model
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
model = DecisionTreeClassifier()
model.fit(X, y)

# Step 3: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved with 10 symptoms!")