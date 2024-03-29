from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open('stuclassifier.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pickle', 'rb'))
except FileNotFoundError:
    print("Error: Model or scaler file not found.")
    exit(1)
except Exception as e:
    print("Error:", e)
    exit(1)

expected_input_length = 17

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input
        int_features = [int(request.form[name]) for name in request.form]
        if len(int_features) != expected_input_length:
            raise ValueError("Incorrect number of input features.")

        # Transform features using scaler
        final_features = scaler.transform([int_features])

        # Make prediction
        prediction = model.predict(final_features)

        # Process prediction result
        output = "Pass" if prediction[0] == 1 else "Fail"

        return render_template('index.html', prediction_text='The student\'s end of year final exam result will be {}'.format(output))

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        print("Error:", e)  # Print the error for debugging
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == "__main__":
    app.run(debug=True)
