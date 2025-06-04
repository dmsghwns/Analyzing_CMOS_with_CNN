from os import system
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# TPU setup
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    tpu = None
    print("TPU not found, falling back to default strategy")

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# Start time measurement
start_time = time.time()
start_monotonic = time.monotonic()

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Optimize dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, tf.one_hot(y_train, 10)))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, tf.one_hot(y_test, 10)))
test_dataset = test_dataset.batch(128).prefetch(tf.data.AUTOTUNE)

# Define model
with strategy.scope():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
        ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  steps_per_execution=32)

# Training
history = model.fit(train_dataset, epochs=200, validation_data=test_dataset)

# Evaluation
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# End time measurement
end_time = time.time()
end_monotonic = time.monotonic()
training_time = end_time - start_time

# Estimate power consumption and energy efficiency
avg_power = 250  # Average power of TPU v3-8 (W)
total_energy_joules = avg_power * training_time
total_energy_kwh = total_energy_joules / 3600000  # Joules to kWh
total_samples = 60000 * 200
energy_per_sample = total_energy_joules / total_samples

# Calculate throughput
throughput = total_samples / training_time

print(f"Training time: {training_time:.2f} seconds")
print(f"Total energy consumption: {total_energy_kwh:.4f} kWh")
print(f"Energy per sample: {energy_per_sample:.4f} Joules/sample")
print(f"Throughput: {throughput:.2f} samples/second")

# Plots
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='x', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='x', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
