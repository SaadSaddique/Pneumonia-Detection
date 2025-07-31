import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Paths
test_dir = "../Dataset/chest_xray/test"
model_path = "../results/best_model.keras"

# Load test dataset
img_size = (150, 150)
batch_size = 32

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

# ✅ Ensure label type is float32 and shape is correct
def cast_labels(x, y):
    y = tf.cast(y, tf.float32)
    y = tf.reshape(y, (-1, 1))  # Make shape (batch_size, 1)
    return x, y

test_ds = test_ds.map(cast_labels)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load model
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded from: {model_path}")

# Evaluate
loss, accuracy = model.evaluate(test_ds)
print(f"\n🧪 Test Accuracy: {accuracy:.4f}")
print(f"🧪 Test Loss: {loss:.4f}")

# Get true labels and predictions
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).flatten()
y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
