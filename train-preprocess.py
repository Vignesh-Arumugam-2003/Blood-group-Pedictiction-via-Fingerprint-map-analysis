import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 103, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))  # Adding one more convolutional layer
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))  # Increase the number of neurons in the fully connected layers
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))

# Compile the model with a reduced learning rate
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators with adjusted data augmentation parameters
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduce rotation range
    width_shift_range=0.1,  # Reduce width shift range
    height_shift_range=0.1,  # Reduce height shift range
    shear_range=0.1,  # Reduce shear range
    zoom_range=0.1,  # Reduce zoom range
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32  # Adjust batch size

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(96, 103),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(96, 103),
    batch_size=batch_size,
    class_mode='categorical'
)

# Adjust steps per epoch based on your dataset size
steps_per_epoch_train = len(train_generator)
steps_per_epoch_val = len(val_generator)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=50,  # Reduce number of epochs
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val
)

model.summary()

# Save the trained model
model.save('model-pre.h5')

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
