import joblib
import pandas as pd

def check_ride_risk(ride_data_dict):
    # load the active operational model and required features
    package = joblib.load('models/active_model.pkl')
    model = package['model']
    required_features = package['features']

    # safety protocol: inject neutral values for any missing features
    for feature in required_features:
        if feature not in ride_data_dict:
            ride_data_dict[feature] = 0.0  

    # flatten to dataframe for inference
    df = pd.DataFrame([ride_data_dict])
    X_live = df[required_features]

    # execute prediction
    prediction = model.predict(X_live)[0]
    probability = model.predict_proba(X_live)[0][1] 
    
    print(f"calculated cancellation probability: {probability * 100:.2f}%")
    
    if prediction == 1:
        return "flag: high risk of cancellation. dispatch alternative."
    else:
        return "clear: ride likely to complete."