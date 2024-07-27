import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Using pandas to read the CSV file
instagram_df_train = pd.read_csv('insta_train.csv')
instagram_df_test = pd.read_csv('insta_test.csv')

# Removing the 'fake' column from the data because it's our output
x_train = instagram_df_train.drop(columns=['fake'])
x_test = instagram_df_test.drop(columns=['fake'])

y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']

scaler_x = StandardScaler()
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

model = tf.keras.models.Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))  # Initial layer of the neuron
# Hidden layers
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
# Output layer
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the scaler
import joblib
joblib.dump(scaler_x, 'scaler_x.pkl')

# Save the model
model.save('instagram_fake_detector_model.h5')
