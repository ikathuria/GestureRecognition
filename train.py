"""The main CNN."""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Step 1: Converting dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    "data2/train",
    target_size=(128, 128),
    batch_size=4,
    color_mode="grayscale",
    class_mode="categorical",
)

test_set = test_datagen.flow_from_directory(
    "data2/test",
    target_size=(128, 128),
    batch_size=4,
    color_mode="grayscale",
    class_mode="categorical",
)

# Step 2: Model
# Conv + Pooling + Fully connected (Dense)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.40))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.40))
model.add(Dense(17))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

# Step 3: Fit the model
model.fit_generator(
    training_set,
    steps_per_epoch=340,  # (80*17)/4
    epochs=100,
    validation_data=test_set,
    validation_steps=51,  # (12*17)/4
)

# Step 4: Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')

model.save_weights('model.h5')
print('Weights saved')
