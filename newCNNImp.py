#-- Amy Butterfield --#
#-- Classical  Layer--#

import os
import glob
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Sequential

# ------------------------------------------------------------------
# GPU setup (must run before any TensorFlow operations)
# ------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPUs available:", gpus)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
SEED = 123
tf.keras.utils.set_random_seed(SEED)

# Images are 512x512 on disk; we load them at 256x256 to save memory
# while keeping more fine detail (hairline fractures) than 128x128.
img_height = 256
img_width = 256
BATCH_SIZE = 16

N_QUBITS = 16      # final feature count handed to the quantum circuit
EPOCHS = 30

AUTOTUNE = tf.data.AUTOTUNE

#training_dir should direct to "BoneFract A Bone Fracture Dataset/BoneFract A Bone Fracture Dataset/train/"
#valid_dir should direct to "BoneFract A Bone Fracture Dataset/BoneFract A Bone Fracture Dataset/valid/"

training_dir = #Add directory
valid_dir = #Add directory

# ------------------------------------------------------------------
# DONE: Input Datasets into a tf.data.Dataset object
# ------------------------------------------------------------------
# Folder layout: <split>/<BodyPart>/<patient>/<Positive|Negative>/<image>.png
# image_dataset_from_directory would only see the body part, so we build the
# dataset ourselves and take the label from the Positive/Negative folder.
class_names = ["Negative", "Positive"]   # label 0 = no fracture, 1 = fracture
num_classes = len(class_names)


def build_dataset(root, shuffle):
    """Returns (dataset, labels, body_parts).
    labels and body_parts are numpy arrays in the same order as the files,
    so they only line up with the dataset when shuffle=False."""
    paths = []
    for ext in ("png", "jpg", "jpeg"):
        paths += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    paths = sorted(paths)

    labels, body_parts = [], []
    for p in paths:
        folder = os.path.basename(os.path.dirname(p)).lower()   # "positive"/"negative"
        assert folder.startswith(("pos", "neg")), f"Unexpected folder for {p}"
        labels.append(1 if folder.startswith("pos") else 0)
        rel = os.path.relpath(p, root)
        body_parts.append(rel.split(os.sep)[0])                 # Arm, Foot, ...

    labels = np.array(labels, dtype=np.int32)
    body_parts = np.array(body_parts)
    print(f"{root}\n  {len(paths)} images: "
          f"{np.sum(labels == 0)} negative, {np.sum(labels == 1)} positive")

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        img.set_shape([None, None, 1])
        img = tf.image.resize(img, (img_height, img_width))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds, labels, body_parts


# Shuffled copy for training; unshuffled copy so extracted features stay
# aligned with their labels and body parts.
training_ds, y_train_all, _ = build_dataset(training_dir, shuffle=True)
train_eval_ds, y_train_ordered, bp_train = build_dataset(training_dir, shuffle=False)
valid_ds, y_valid_ordered, bp_valid = build_dataset(valid_dir, shuffle=False)

print("Classes:", class_names)

# Class weights, so an imbalance between Positive and Negative doesn't let the
# model score well by always guessing the bigger class.
counts = np.bincount(y_train_all, minlength=num_classes)
class_weight = {i: len(y_train_all) / (num_classes * counts[i]) for i in range(num_classes)}
print("Class weights:", class_weight)

# Check for a confounder: does fracture rate depend heavily on body part?
for bp in np.unique(bp_train):
    m = bp_train == bp
    print(f"  {bp:>10}: {m.sum():6d} images, {100 * y_train_ordered[m].mean():.1f}% positive")

# ------------------------------------------------------------------
# Convolute this to a manageable size for a qBraid computer
# ------------------------------------------------------------------
# Normalize pixel values to [0, 1]
preprocess = Sequential([
    layers.Rescaling(1. / 255),
], name="preprocess")

# Light augmentation, active only during training (never at feature extraction)
augment = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augment")

classical_layer = Sequential([
    layers.Input(shape=(img_height, img_width, 1)),
    layers.Conv2D(8, 5, padding="same", activation="relu"),
    layers.MaxPooling2D(4),           # 256 -> 64
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(4),           # 64 -> 16
    layers.Conv2D(8, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(2),           # 16 -> 8  (8x8x8 = 512 values)
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(N_QUBITS),           # no activation here
    layers.BatchNormalization(),      # rescales each feature to ~unit spread
    layers.Activation("tanh"),        # lands in the sensitive middle range, not saturated
], name="classical_layer")

# ------------------------------------------------------------------
# Step 1: pre-train the classical layer with a throwaway classifier head
# ------------------------------------------------------------------
head = layers.Dense(num_classes)      # logits

full_model = Sequential([preprocess, augment, classical_layer, head])
full_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
full_model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=7, restore_best_weights=True
    )
]

history = full_model.fit(
    training_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
)

# Training curves (saved with a unique name, shown without blocking the script)
run_id = int(time.time())
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="valid acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f"training_curves {run_id}.png", dpi=150)
plt.show(block=False)

# ------------------------------------------------------------------
# Step 2: freeze the classical layer and extract quantum-ready features
# ------------------------------------------------------------------
classical_layer.trainable = False


def extract_features(ds):
    """Run every image through preprocess + classical_layer.
    Returns features as a numpy array of shape (num_images, N_QUBITS).
    Features are scaled from tanh's [-1, 1] to [-pi/2, pi/2] so that +/- extremes
    can never map to the same rotation angle."""
    feats = []
    for x, _ in ds:
        f = classical_layer(preprocess(x), training=False) * (np.pi / 2)
        feats.append(f.numpy())
    return np.concatenate(feats)


X_train = extract_features(train_eval_ds)
X_valid = extract_features(valid_ds)
y_train, y_valid = y_train_ordered, y_valid_ordered

print("Train features:", X_train.shape)   # (num_images, N_QUBITS)
print("Valid features:", X_valid.shape)
print("Feature std per qubit (healthy is roughly 0.3-1.0):",
      np.round(X_train.std(axis=0), 2))

# ------------------------------------------------------------------
# Accuracy per body part on the validation set (confounder check)
# ------------------------------------------------------------------
logits = full_model.predict(valid_ds, verbose=0)
preds = np.argmax(logits, axis=1)
print("Overall valid accuracy: %.3f" % np.mean(preds == y_valid))
for bp in np.unique(bp_valid):
    m = bp_valid == bp
    print(f"  {bp:>10}: {100 * np.mean(preds[m] == y_valid[m]):.1f}% ({m.sum()} images)")

# ------------------------------------------------------------------
# Step 3: save for the quantum layer (upload this small file to qBraid)
# ------------------------------------------------------------------
out_name = f"quantum_features {run_id}.npz"
np.savez(
    out_name,
    X_train=X_train, y_train=y_train,
    X_valid=X_valid, y_valid=y_valid,
    body_part_train=bp_train, body_part_valid=bp_valid,
    class_names=np.array(class_names),
)
print("Saved", out_name)

# The .npz is tiny compared to the images, so it's easy to import on qBraid:
#   data = np.load("quantum_features <id>.npz")
#   X_train, y_train = data["X_train"], data["y_train"]

plt.show()   # keeps the plot window open at the end until you close it