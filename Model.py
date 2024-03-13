import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import keras
from keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Flatten
from keras.models import Model
from keras.metrics import Precision, Recall, AUC
from keras.regularizers import l2

dataset = pd.read_csv("CCD.csv")

labels = dataset['default payment next month'].values
features = dataset.drop('default payment next month', axis=1)

########################################################
#################### PRE-PROCESSING ####################
########################################################

categorical_feature_names = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

PAY_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
features[PAY_columns] = features[PAY_columns] + 2 # Add 2 to PAY columns, for embedding layer

features['SEX'] = features['SEX'] - 1 # bring from [1,2] -> [0,1] for embedding layer

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=42) # 0.18*0.85=0.15

##### SMOTE
sm = SMOTENC(categorical_features=categorical_feature_names,random_state=42, sampling_strategy=0.8)
X_train, y_train = sm.fit_resample(X_train, y_train)

##### Normalization and feature splitting
sc_bill = StandardScaler()
X_train_bill_amt = sc_bill.fit_transform(X_train[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])
X_test_bill_amt = sc_bill.transform(X_test[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])
X_val_bill_amt = sc_bill.transform(X_val[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']])

sc_pay = StandardScaler()
X_train_pay_amt = sc_pay.fit_transform(X_train[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])
X_test_pay_amt = sc_pay.transform(X_test[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])
X_val_pay_amt = sc_pay.transform(X_val[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']])

X_train_pay = X_train[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
X_test_pay = X_test[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
X_val_pay = X_val[['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

sc = StandardScaler()
X_train_static_continuous = sc.fit_transform(X_train[['LIMIT_BAL', 'AGE']])
X_test_static_continuous = sc.transform(X_test[['LIMIT_BAL', 'AGE']])
X_val_static_continuous = sc.transform(X_val[['LIMIT_BAL', 'AGE']])

X_train_sex = X_train['SEX']
X_test_sex = X_test['SEX']
X_val_sex = X_val['SEX']

X_train_education = X_train['EDUCATION']
X_test_education = X_test['EDUCATION']
X_val_education = X_val['EDUCATION']

X_train_marriage = X_train['MARRIAGE']
X_test_marriage = X_test['MARRIAGE']
X_val_marriage = X_val['MARRIAGE']


#############################################
################### MODEL ###################
#############################################

### Static Features
static_input = Input(shape=(2,), name='static')
sex_input = Input(shape=(1,), name='sex')
education_input = Input(shape=(1,), name='education')
marriage_input = Input(shape=(1,), name='marriage')

sex_embedding = Flatten()(Embedding(input_dim=1+1, output_dim=5)(sex_input))
education_embedding = Flatten()(Embedding(input_dim=6+1, output_dim=5)(education_input))
marriage_embedding = Flatten()(Embedding(input_dim=3+1, output_dim=5)(marriage_input))

static = Concatenate()([static_input, sex_embedding, education_embedding, marriage_embedding])

### PAY Features
pay_input = Input(shape=(6,), name='pay')
pay = Embedding(input_dim=10 + 1, output_dim=6)(pay_input)
pay = LSTM(64, return_sequences=True)(pay)
pay = LSTM(32)(pay)

### BILL_AMT Features
bill_amt_input = Input(shape=(6, 1), name='bill_amt')
bill_amt = LSTM(64, return_sequences=True)(bill_amt_input)
bill_amt = LSTM(32)(bill_amt)

### PAY_AMT Features
pay_amt_input = Input(shape=(6, 1), name='pay_amt')
pay_amt = LSTM(64, return_sequences=True)(pay_amt_input)
pay_amt = LSTM(32)(pay_amt)

### Combine and Dense layers
x = Concatenate()([pay, bill_amt, pay_amt, static])
x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
output = Dense(1, activation='sigmoid')(x)

inputs_list = [static_input, sex_input, education_input, marriage_input, pay_input, bill_amt_input, pay_amt_input]

train_inputs = [X_train_static_continuous, X_train_sex, X_train_education, X_train_marriage, 
                X_train_pay, X_train_bill_amt, X_train_pay_amt]

val_inputs = [X_val_static_continuous, X_val_sex, X_val_education, X_val_marriage,
              X_val_pay, X_val_bill_amt, X_val_pay_amt]

test_inputs = [X_test_static_continuous, X_test_sex, X_test_education, X_test_marriage, 
                X_test_pay, X_test_bill_amt, X_test_pay_amt]

model = Model(inputs=inputs_list, outputs=output)
model.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
             )

history = model.fit(
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),
    epochs=25,
    batch_size=64,
)

evaluation = model.evaluate(
    test_inputs, y_test, verbose=1)

print('Test Loss:', evaluation[0])
print('Test Accuracy:', evaluation[1])
print('Test Precision:', evaluation[2])
print('Test Recall:', evaluation[3])
print('Test AUC:', evaluation[4])


# Plot Accuracy
plt.figure(figsize=(7, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# Plot Precision
plt.figure(figsize=(7, 4))
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.title('Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# Plot Recall
plt.figure(figsize=(7, 4))
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.title('Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# Plot Loss
plt.figure(figsize=(7, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.show()