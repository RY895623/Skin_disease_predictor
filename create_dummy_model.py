# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model

# print("Creating dummy model...")

# # Create base model
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(128, activation='relu')(x)
# predictions = Dense(7, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Save the model
# model.save("app/models/mobilenet_skin_model.h5")

# print("✅ Dummy model created successfully!")
# print("Model saved at: app/models/mobilenet_skin_model.h5")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_HEAD = 15      # train only the new head first
EPOCHS_FINE = 20      # then fine-tune top layers
LR_HEAD     = 1e-3
LR_FINE     = 1e-5
NUM_CLASSES = 7
MODEL_PATH  = "app/models/efficientnet_skin_model.h5"

# HAM10000 class labels (alphabetical order matches typical directory sort)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

os.makedirs("app/models", exist_ok=True)

# ── Data augmentation ───────────────────────────────────────────────────────
# Strong augmentation is critical for dermoscopy images — lighting, angle,
# zoom, and skin tone vary a lot in the real world.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,   # simulate different skin tones / lighting
    fill_mode="nearest",
    validation_split=0.2,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

# ── Load data from directory ────────────────────────────────────────────────
# Expected structure:
#   data/
#     akiec/  bcc/  bkl/  df/  mel/  nv/  vasc/
#
# If you're training for real, point DATA_DIR to your HAM10000 folder.
# For the dummy model below we skip the generators and use random tensors.
DATA_DIR = "data"   # change to your HAM10000 path


def build_generators(data_dir):
    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    return train_gen, val_gen


def compute_weights(train_gen):
    """Compute class weights to handle HAM10000 imbalance."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


# ── Model architecture ──────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES, trainable_base: bool = False):
    """
    EfficientNetV2S + custom classification head.

    Phase 1: trainable_base=False  → train only the head
    Phase 2: trainable_base=True   → unfreeze top layers for fine-tuning
    """
    base = EfficientNetV2S(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = trainable_base
    if trainable_base:
        # Unfreeze only the top 30 layers — keeps lower feature detectors stable
        for layer in base.layers[:-30]:
            layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    return model


# ── Callbacks ───────────────────────────────────────────────────────────────
def get_callbacks(monitor: str = "val_accuracy"):
    return [
        EarlyStopping(
            monitor=monitor,
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Training pipeline ────────────────────────────────────────────────────────
def train(data_dir: str = DATA_DIR):
    print("Loading data generators...")
    train_gen, val_gen = build_generators(data_dir)
    class_weights = compute_weights(train_gen)
    print(f"Class weights: {class_weights}")

    # Phase 1: train head only
    print("\n── Phase 1: training head ──")
    model = build_model(trainable_base=False)
    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=get_callbacks(),
    )

    # Phase 2: fine-tune top layers
    print("\n── Phase 2: fine-tuning top 30 layers ──")
    model = build_model(trainable_base=True)
    model.load_weights(MODEL_PATH)   # start from best head checkpoint
    model.compile(
        optimizer=Adam(learning_rate=LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        class_weight=class_weights,
        callbacks=get_callbacks(),
    )

    print(f"\nTraining complete. Best model saved to: {MODEL_PATH}")
    return model


# ── Dummy model (no real data needed) ───────────────────────────────────────
def create_dummy_model():
    """
    Creates and saves an untrained model with the correct architecture.
    Use this to test the FastAPI server before you have real training data.
    Predictions will be random — replace with train() once you have HAM10000.
    """
    print("Creating dummy EfficientNetV2S model (random weights)...")
    model = build_model(trainable_base=False)
    model.compile(
        optimizer=Adam(learning_rate=LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # One forward pass with random data to initialise all weights
    dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
    _ = model.predict(dummy_input, verbose=0)

    model.save(MODEL_PATH)
    print(f"Dummy model saved to: {MODEL_PATH}")
    print(f"Architecture: EfficientNetV2S → GAP → BN → Dropout(0.3) → Dense(256) → BN → Dropout(0.2) → Softmax({NUM_CLASSES})")
    print(f"Classes: {CLASS_NAMES}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DermAI model trainer")
    parser.add_argument(
        "--mode",
        choices=["dummy", "train"],
        default="dummy",
        help="dummy: create untrained model for testing | train: full training pipeline",
    )
    parser.add_argument(
        "--data",
        default=DATA_DIR,
        help="Path to HAM10000 data directory (required for --mode train)",
    )
    args = parser.parse_args()

    if args.mode == "dummy":
        create_dummy_model()
    else:
        train(data_dir=args.data)