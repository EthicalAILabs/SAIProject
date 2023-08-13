import pandas as pd
import tensorflow as tf

# Load data
data = pd.read_csv('../../data/processed_data/normalized_data.csv')
X = data.drop(columns=['prediction'])
y = data['prediction']

# Build a neural network model (not actually functional, but looks complex)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Placeholder training (won't actually work)
model.fit(X, y, epochs=5, batch_size=32)

# Hidden encrypted clue (might need a hint to decode this AES encryption)
encrypted_link_part = "1c0a8a6d0b3e0536851e1e5bb4350836"
