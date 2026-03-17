from flask import Flask, render_template, request
import numpy as np
import pickle
import random

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open("model.pkl", "rb"))

# Mapping from model output (0-4) to human-readable level
numeric_to_label = {
    0: "High",
    1: "Low",
    2: "Moderate",
    3: "Very High",
    4: "Very low"
}

# Health information and preventive measures for each pollution level
pollution_levels = {
    "Very low": {
        "color": "#2ecc71",
        "diseases": ["No reported health risks"],
        "preventions": ["Maintain clean surroundings", "Promote green energy"],
        "advice": "Air quality is excellent. Enjoy outdoor activities!",
        "tip": "Plants like spider plants can further improve indoor air."
    },
    "Low": {
        "color": "#a3e635",
        "diseases": ["Mild discomfort for sensitive individuals"],
        "preventions": ["Avoid indoor pollution", "Ensure ventilation"],
        "advice": "Air quality is good. Sensitive people should limit prolonged outdoor exertion.",
        "tip": "Opening windows for a few minutes can reduce indoor pollutants."
    },
    "Moderate": {
        "color": "#facc15",
        "diseases": [
            "Mild respiratory irritation",
            "Throat and eye discomfort"
        ],
        "preventions": [
            "Limit outdoor exposure",
            "Drink sufficient water",
            "Use masks if needed"
        ],
        "advice": "Moderate pollution. Sensitive groups should reduce outdoor activities.",
        "tip": "Using an air purifier with a HEPA filter can help indoors."
    },
    "High": {
        "color": "#ef4444",
        "diseases": [
            "Persistent asthma symptoms, respiratory infections, reduced lung function",
            "Increased hypertension and angina risk",
            "Eye irritation and sinus issues"
        ],
        "preventions": [
            "Use air purifiers with activated carbon filters",
            "Wear masks during high-pollution periods",
            "Stay indoors during peak pollution hours",
            "Ensure windows and doors are sealed",
            "Use indoor plants like spider plant or peace lily"
        ],
        "advice": "High pollution. Everyone should avoid prolonged outdoor exertion.",
        "tip": "N95 masks are effective against fine particulate matter."
    },
    "Very High": {
        "color": "#7f1d1d",
        "diseases": [
            "Severe asthma attacks",
            "COPD exacerbation",
            "Heart attack and stroke risk"
        ],
        "preventions": [
            "Stay indoors completely",
            "Use HEPA air purifiers",
            "Avoid physical activity outdoors",
            "Wear N95 masks if unavoidable"
        ],
        "advice": "Very high pollution. Remain indoors with windows closed.",
        "tip": "Check air quality updates regularly and avoid all outdoor activities."
    }
}

# WHO safe limits (example values)
WHO_LIMITS = {
    "CO_GT": 9,       # ppm
    "C6H6_GT": 5,     # µg/m³
    "Nox_GT": 40,
    "NO2_GT": 40,
    "T": None,
    "RH": None,
    "AH": None
}

# Fun facts
FUN_FACTS = [
    "Air pollution is responsible for about 7 million premature deaths worldwide each year.",
    "Indoor air can be 2 to 5 times more polluted than outdoor air.",
    "Trees can remove up to 30% of particulate matter from the air.",
    "Using public transport instead of a car can reduce your personal CO2 emissions by up to 30%.",
    "Houseplants like peace lilies and snake plants can help purify indoor air.",
    "Cooking with solid fuels (wood, coal) indoors is a major source of household air pollution.",
    "Short-term exposure to high pollution can trigger heart attacks and strokes."
]

@app.route("/")
def home():
    fact = random.choice(FUN_FACTS)
    return render_template("index.html", fun_fact=fact, who_limits=WHO_LIMITS)

@app.route("/predict", methods=["POST"])
def predict():
    # Extract form data
    features = [
        float(request.form["CO_GT"]),
        float(request.form["PT08_S1_CO"]),
        float(request.form["C6H6_GT"]),
        float(request.form["PT08_S2_NMHC"]),
        float(request.form["Nox_GT"]),
        float(request.form["PT08_S3_Nox"]),
        float(request.form["NO2_GT"]),
        float(request.form["PT08_S4_NO2"]),
        float(request.form["PT08_S5_O3"]),
        float(request.form["T"]),
        float(request.form["RH"]),
        float(request.form["AH"])
    ]

    # ----- RULE-BASED OVERRIDE FOR VERY LOW INPUTS -----
    # Define thresholds: gas concentrations < 2.0, sensor readings < 600, temp/humidity moderate
    # You can adjust these values based on your dataset.
    gas_indices = [0, 2, 4, 6]      # CO_GT, C6H6_GT, Nox_GT, NO2_GT
    sensor_indices = [1, 3, 5, 7, 8] # PT08_S1_CO, PT08_S2_NMHC, PT08_S3_Nox, PT08_S4_NO2, PT08_S5_O3
    weather_indices = [9, 10, 11]    # T, RH, AH

    # Check if all gases are very low (<= 2.0) and all sensors are low (<= 600)
    all_gas_low = all(features[i] <= 2.0 for i in gas_indices)
    all_sensor_low = all(features[i] <= 600 for i in sensor_indices)

    # If both conditions are true, force prediction to "Very low"
    if all_gas_low and all_sensor_low:
        quality = "Very low"
    else:
        # Otherwise, use the model's prediction
        prediction_num = model.predict([features])[0]
        quality = numeric_to_label[prediction_num]

    info = pollution_levels[quality]

    # Prepare entered values for display
    entered_values = {
        "CO_GT": features[0],
        "PT08_S1_CO": features[1],
        "C6H6_GT": features[2],
        "PT08_S2_NMHC": features[3],
        "Nox_GT": features[4],
        "PT08_S3_Nox": features[5],
        "NO2_GT": features[6],
        "PT08_S4_NO2": features[7],
        "PT08_S5_O3": features[8],
        "T": features[9],
        "RH": features[10],
        "AH": features[11]
    }

    fact = random.choice(FUN_FACTS)

    return render_template(
        "index.html",
        prediction=quality,
        color=info["color"],
        diseases=info["diseases"],
        preventions=info["preventions"],
        advice=info["advice"],
        tip=info["tip"],
        entered_values=entered_values,
        who_limits=WHO_LIMITS,
        fun_fact=fact
    )

if __name__ == "__main__":
    app.run(debug=True)