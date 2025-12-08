# ResNet50_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# ---------------- CONFIG ----------------
DATASET_DIR = 'Pet_Dataset'  # Folder containing 8 class subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8  # initial training before fine-tuning
NUM_CLASSES = 8  # Angry, Happy, Sad, Scared, Surprised, Relaxed, Alert, Hungry

# ---------------- DATA GENERATORS ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# ---------------- BUILD MODEL ----------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze base model initially
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------- TRAIN MODEL (initial phase) ----------------
print("\n🔹 Starting initial training (frozen base)...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

# ---------------- FINE-TUNING ----------------
print("\n🔹 Starting fine-tuning (unfreezing top layers)...\n")

# Unfreeze last 50 layers for better learning
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compile again with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training (fine-tuning)
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,  # fine-tuning epochs
    verbose=1
)

# ---------------- SAVE MODEL ----------------
model.save('pet_emotion_resnet50.h5')
print("✅ Model saved successfully as pet_emotion_resnet50.h5")
