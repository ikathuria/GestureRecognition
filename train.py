"""The main CNN."""

from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
import os
from image_processing import DataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Step 1: Converting dataset
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False
# )

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# training_set = train_datagen.flow_from_directory(
#     "data2/train",
#     target_size=(210, 280),
#     batch_size=4,
#     color_mode="grayscale",
#     class_mode="categorical",
# )

# test_set = test_datagen.flow_from_directory(
#     "data2/test",
#     target_size=(210, 280),
#     batch_size=4,
#     color_mode="grayscale",
#     class_mode="categorical",
# )

training_generator = DataGenerator('train')
validation_generator = DataGenerator('val')

# Step 2: Model
# Conv + Pooling + Fully connected (Dense)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(210, 280, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.20))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.20))
model.add(Dense(17))

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# Step 3: Fit the model
model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    epochs=1
)

# Step 4: Save the model
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')

model.save_weights('model/model.h5')
print('Weights saved')
