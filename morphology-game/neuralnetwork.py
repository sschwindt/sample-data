import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# load dataset
filename = 'XS-morphology.csv'
data = pd.read_csv(filename, sep=',')

# preprocess data
X = data[['W', 'S', 'Q', 'U', 'H']].values
y = data['Morphology'].values

# encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)

# evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy: ", accuracy)

# calculate and print F1 score
f1 = f1_score(y_test, y_pred_classes, average='weighted')
print("F1 Score: ", f1)

# print classification report
print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
