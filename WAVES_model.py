import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

def haar_wavelet_tf(input_img):
    # Define Haar filters
    h0 = tf.constant([[1, 1]], dtype=tf.float32) / 2.0
    h1 = tf.constant([[1, -1]], dtype=tf.float32) / 2.0

    # Tile filters for 3 RGB channels
    h0 = tf.tile(h0, [1, 3])
    h1 = tf.tile(h1, [1, 3])

    # Expand dims for conv2d to handle RGB channels
    h0 = tf.reshape(h0, [2, 1, 3, 1])
    h1 = tf.reshape(h1, [2, 1, 3, 1])

    # Low pass and high pass filters
    img_L = tf.nn.conv2d(input_img, h0, strides=[1, 2, 1, 1], padding='SAME')
    img_H = tf.nn.conv2d(input_img, h1, strides=[1, 2, 1, 1], padding='SAME')

    # Concatenate results
    return tf.concat([img_L, img_H], axis=2)


def haar_wavelet_2D_tf(image):
    rows = haar_wavelet_tf(image)
    transposed = tf.transpose(image, perm=[0, 2, 1, 3])
    cols = haar_wavelet_tf(transposed)
    return tf.transpose(cols, perm=[0, 2, 1, 3])

def cnn_feature_extractor(input_tensor):
    x_cnn = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
    x_cnn = tf.keras.layers.MaxPooling2D((2, 2))(x_cnn)
    x_cnn = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x_cnn)
    x_cnn = tf.keras.layers.MaxPooling2D((2, 2))(x_cnn)
    x_cnn = tf.keras.layers.Flatten()(x_cnn)
    return x_cnn


input_img = tf.keras.layers.Input(shape=(224, 224, 3))

haar_output = haar_wavelet_2D_tf(input_img)
cnn_features = cnn_feature_extractor(haar_output)

# Combine CNN features and global features obtained from GlobalMaxpooling1d layer of SPADE.
combined_features = tf.keras.layers.Concatenate()([global_features, cnn_features])

# MLP head
x_mlp = tf.keras.layers.Dense(128, activation='relu')(combined_features)
x_mlp = tf.keras.layers.Dropout(0.5)(x_mlp)
output = tf.keras.layers.Dense(3, activation='softmax')(x_mlp)

model = tf.keras.models.Model(inputs=[inputs,input_img], outputs=output)

model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

batch_size = 32
data_generator = MultiModalDataGenerator(train_points, train_images, train_labels, batch_size)
test_data_generator = MultiModalDataGenerator(test_points, test_images, test_labels, batch_size)
earlystopping = EarlyStopping(monitor='loss', min_delta = 0, patience = 30, verbose = 1, restore_best_weights=True)
model.fit(data_generator, epochs=200, callbacks=[earlystopping])
