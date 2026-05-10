import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the data
print("Loading data... please wait.")
df = pd.read_csv('New credit card records.csv')

# 2. Scale the 'Amount' column
# AI works better when numbers are in a similar small range
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# 3. Separate features from the result
# X is the data (V1-V28, Amount), y is the answer (0 for safe, 1 for fraud)
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# 4. Split data for testing (80% to learn, 20% to test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and Train the 'Brain'
print("Training the AI model... this might take 30 seconds.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. SAVE the files so we can use them later
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("--- DONE ---")
print("I have created 'fraud_detection_model.pkl' and 'scaler.pkl' in your folder!")