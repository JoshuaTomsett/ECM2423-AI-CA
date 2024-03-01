## Added SMOTE / second hidden layer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, AUC

# Load the dataset
dataset = pd.read_csv("CCD.csv")

# Labels
labels = dataset['default payment next month'].values

# Drop the target column to only include features
features = dataset.drop('default payment next month', axis=1)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Normalize the time series and static features separately to preserve the temporal structure
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_train_pay = scaler.fit_transform(X_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']])
X_test_pay = scaler.transform(X_test[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']])

X_train_bill = scaler.fit_transform(X_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])
X_test_bill = scaler.transform(X_test[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])

X_train_pay_amt = scaler.fit_transform(X_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])
X_test_pay_amt = scaler.transform(X_test[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])

# Normalize static features
static_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'LIMIT_BAL']
X_train_static = scaler.fit_transform(X_train[static_features])
X_test_static = scaler.transform(X_test[static_features])

# Reshape the series data for LSTM input
X_train_pay = X_train_pay.reshape(-1, 6, 1)
X_test_pay = X_test_pay.reshape(-1, 6, 1)

X_train_bill = X_train_bill.reshape(-1, 6, 1)
X_test_bill = X_test_bill.reshape(-1, 6, 1)

X_train_pay_amt = X_train_pay_amt.reshape(-1, 6, 1)
X_test_pay_amt = X_test_pay_amt.reshape(-1, 6, 1)

# Time series inputs
pay_input = Input(shape=(6, 1), name='pay_input')
bill_amt_input = Input(shape=(6, 1), name='bill_amt_input')
pay_amt_input = Input(shape=(6, 1), name='pay_amt_input')

# Static input
static_input = Input(shape=(X_train_static.shape[1],), name='static_input')

# LSTM layers for each time series input
pay_features = LSTM(32)(pay_input)
bill_amt_features = LSTM(32)(bill_amt_input)
pay_amt_features = LSTM(32)(pay_amt_input)

# Concatenate all the features
combined_features = Concatenate()([pay_features, bill_amt_features, pay_amt_features, static_input])

# Hidden layer
x = Dense(64, activation='sigmoid')(combined_features)
y = Dense(64, activation='sigmoid')(x)

# Output layer
output = Dense(1, activation='sigmoid')(y)

# Create the model
model = Model(inputs=[pay_input, bill_amt_input, pay_amt_input, static_input], outputs=output)


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
             )

# model.summary()

history = model.fit(
    [X_train_pay, X_train_bill, X_train_pay_amt, X_train_static], y_train,
    validation_data=([X_test_pay, X_test_bill, X_test_pay_amt, X_test_static], y_test),
    epochs=10,
    batch_size=32
)

# After training, if you want to evaluate the model on the test set, you can use:
evaluation = model.evaluate(
    [X_test_pay, X_test_bill, X_test_pay_amt, X_test_static], y_test, verbose=1
)

# Updated to print all evaluation metrics
print('Test Loss:', evaluation[0])
print('Test Accuracy:', evaluation[1])
print('Test Precision:', evaluation[2])
print('Test Recall:', evaluation[3])
print('Test AUC:', evaluation[4])
