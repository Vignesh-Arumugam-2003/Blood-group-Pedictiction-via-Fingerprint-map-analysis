import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 103, 3)))  # Adjust input shape
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))  # Change output units to 8 for 8 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create data generators for training and validation sets without preprocessing
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Adjust batch size and steps per epoch based on your dataset size
batch_size = 16  # Set the batch size according to your dataset size
steps_per_epoch_train = 452 // batch_size  # Total number of images in training set / batch size
steps_per_epoch_val = 113 // batch_size  # Total number of images in validation set / batch size

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(96, 103),  # Resize images to 96x103
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(96, 103),
    batch_size=batch_size,
    class_mode='categorical'
)

# Get the class labels for the training set
train_class_labels = train_generator.class_indices
print("Class Labels (Training):", train_class_labels)

# Get the class labels for the validation set
val_class_labels = val_generator.class_indices
print("Class Labels (Validation):", val_class_labels)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=10,
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val
)

model.summary()

# Save the trained model
model.save('model-nopre.h5')


# Summarize history for accuracy and loss
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot training and validation accuracy
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot training and validation loss
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()

# Plot a pie chart for class distribution in the training set
fig, ax = plt.subplots()
train_class_counts = train_generator.classes
class_labels = list(train_class_labels.keys())
class_counts = [np.sum(train_class_counts == train_class_labels[label]) for label in class_labels]

ax.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Class Distribution in Training Set')
plt.show()

# Plot a bar chart for the number of images in each class in the validation set
fig, ax = plt.subplots()

val_class_counts = val_generator.classes
class_labels = list(val_class_labels.keys())
class_counts = [np.sum(val_class_counts == val_class_labels[label]) for label in class_labels]

ax.bar(class_labels, class_counts, color='skyblue')
ax.set_title('Number of Images in Each Class (Validation Set)')
ax.set_xlabel('Class')
ax.set_ylabel('Number of Images')

plt.show()
