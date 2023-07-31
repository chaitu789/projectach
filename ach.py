import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

df = pd.read_csv('//home//chaitanya//ach//tourism2.csv')
model = joblib.load('/home//chaitanya/ach/model.joblib')
vectorizer = joblib.load('/home//chaitanya/ach/vectorizer.joblib')


def predict_travel_destinations(region_input, type_input):
    example_input = f"{region_input.strip()} {type_input.strip()}"
    example_input_vectorized = vectorizer.transform([example_input])
    proba = model.predict_proba(example_input_vectorized)[0]

    N = 5  # Number of top predictions you want
    top_N_indices = proba.argsort()[-N:][::-1]
    predicted_destinations = [model.classes_[idx] for idx in top_N_indices]

    return predicted_destinations

def get_location_info(location_name):
    location_info = df[df['Name'] == location_name]
    return location_info.iloc[0]


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        region_input = request.form.get('Region', '')
        type_input = request.form.get('Travel Type', '')
        predicted_destinations = predict_travel_destinations(region_input, type_input)

        # Prepare data for rendering in the template
        destinations_data = []
        for i, destination in enumerate(predicted_destinations, start=1):
            location_info = get_location_info(destination)
            destinations_data.append({
                'number': i,
                'destination': destination,
                'region': location_info['Region'],
                'type': location_info['Type'],
                'airport_dist': location_info['airport_dist(km)'],
                'railway_dist': location_info['railway_dist(km)'],
                'image': location_info['images'],
                'description': location_info['description']
            })

        return render_template('result.html', destinations=destinations_data)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
