import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Check GPU configuration (A100)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Running on GPU: {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU not found, falling back to CPU")

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
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(128).cache().prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, tf.one_hot(y_test, 10)))
test_dataset = test_dataset.batch(128).cache().prefetch(tf.data.AUTOTUNE)

# Define model
with tf.device('/GPU:0'):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Model summary
model.summary()

# Set early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=test_dataset,
                    callbacks=[early_stopping],
                    verbose=1)

# Evaluation
loss, accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# End time measurement
end_time = time.time()
end_monotonic = time.monotonic()
training_time = end_time - start_time
training_time_ns = training_time * 1e9  # Seconds to nanoseconds

# Power consumption and energy calculation (A100 basis, mW)
avg_power_mw = 400000  # 400W = 400,000 mW
total_energy_joules = (avg_power_mw / 1000) * training_time
total_energy_kwh = total_energy_joules / 3600000
total_samples = 60000 * len(history.history['loss'])
energy_per_sample = total_energy_joules / total_samples

# TOPS calculation
def calculate_flops(model, input_shape, batch_size):
    """ Calculate FLOPS (for Conv2D and Dense layers, based on 2 FLOPS per MAC) """
    flops = 0
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            output_shape = layer.output.shape[1:3]  # (height, width)
            input_channels = layer.input.shape[-1]
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
            flops += (kernel_size * layer.filters * input_channels *
                      output_shape[0] * output_shape[1] * batch_size)
        elif isinstance(layer, Dense):
            input_units = layer.input.shape[-1]
            output_units = layer.units
            flops += (input_units * output_units * batch_size)
    return flops * 2  # Multiply-Accumulate

batch_size = 128
flops_per_sample = calculate_flops(model, (28, 28, 1), 1)
total_flops = flops_per_sample * total_samples
tops = total_flops / training_time / 1e12  # TOPS
tops_per_watt = tops / (avg_power_mw / 1000)  # TOPS/W

# Throughput (GOPS)
throughput_gops = total_flops / training_time / 1e9  # GOPS

print(f"Training time: {training_time_ns:.2f} ns")
print(f"Average power consumption: {avg_power_mw:.0f} mW")
print(f"Total energy consumption: {total_energy_kwh:.4f} kWh")
print(f"Energy per sample: {energy_per_sample:.4f} Joules/sample")
print(f"Throughput: {throughput_gops:.2f} GOPS")
print(f"Energy efficiency: {tops_per_watt:.4f} TOPS/W")

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
