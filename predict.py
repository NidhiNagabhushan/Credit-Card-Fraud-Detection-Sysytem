import joblib
import pandas as pd

# 1. Load the AI Brain and the Translator
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Create a "Fake" New Transaction to check
# (We use 29 numbers because the AI expects V1-V28 + Amount)
new_data = [[0]*29] 
new_data[0][28] = 15000.00 # Set the 'Amount' to $500

# 3. Translate the amount and Predict
new_data[0][28] = scaler.transform([[new_data[0][28]]])[0][0]
prediction = model.predict(new_data)

# 4. Show the result
if prediction[0] == 1:
    print("⚠️ ALERT: This transaction looks like FRAUD!")
else:
    print("✅ SUCCESS: This transaction is SAFE.")