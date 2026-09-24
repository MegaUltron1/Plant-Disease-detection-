import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==================================================
# Dataset Path
# ==================================================
DATASET_PATH = r"C:\Users\ashus\Desktop\Plant disease Detection\dataset"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# ==================================================
# Data Preprocessing
# ==================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

valid_data = valid_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "valid"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# Verify class ordering across splits
print("\nClass Indices:")
print("train:", train_data.class_indices)
print("valid:", valid_data.class_indices)

_test_datagen_tmp = ImageDataGenerator(preprocessing_function=preprocess_input)
_test_data_tmp = _test_datagen_tmp.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)
print("test:", _test_data_tmp.class_indices)

assert (
    train_data.class_indices == valid_data.class_indices == _test_data_tmp.class_indices
), "class_indices mismatch between train/valid/test. Check dataset folder structure."

# ==================================================
# Build EfficientNetB0 Model
# ==================================================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
)

# Freeze pretrained layers (train classifier head first)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.35)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.25)(x)

predictions = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ==================================================
# Compile (Stage 1)
# ==================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ==================================================
# Create Models Folder
# ==================================================
os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/crop_disease_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1,
)

# ==================================================
# Train (Stage 1)
# ==================================================
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=30,
    callbacks=[checkpoint, early_stop, reduce_lr],
)

# ==================================================
# Fine-Tuning (Stage 2)
# ==================================================
print("\nStarting Fine-Tuning...\n")

base_model.trainable = True

# Keep most layers frozen for stability during fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    callbacks=[checkpoint, early_stop, reduce_lr],
)

# ==================================================
# Save Final Model
# ==================================================
model.save("models/final_model.keras")

print("\nTraining Completed Successfully!")

# ==================================================
# Combine History + Plots
# ==================================================
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ==================================================
# Model Evaluation on Test Dataset
# ==================================================

# Load Test Dataset (same preprocessing as train/valid)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

predictions = model.predict(test_data)

y_pred = predictions.argmax(axis=1)
y_true = test_data.classes

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\n==========================================")
print("        MODEL EVALUATION METRICS")
print("==========================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(test_data.class_indices.keys()),
    )
)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(test_data.class_indices.keys()),
    yticklabels=list(test_data.class_indices.keys()),
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

