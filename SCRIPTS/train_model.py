import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories
TRAIN_DIR = "DATA/processed_images/"

# Model Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Data Preprocessing
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_data = datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE, class_mode='categorical', subset='training')

val_data = datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE,
                                       batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data.class_indices), activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save Model
model.save("OUTPUT/face_recognition_model.h5")
