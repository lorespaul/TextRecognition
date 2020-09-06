import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from os import path, sys

from FilePaths import FilePaths
from DataLoader import DataLoader, Batch
from WordsLoaderDataset import WordsLoaderDataset
from SamplePreprocessor import preprocess


def build_model():
    model = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 32)),
        keras.layers.Reshape((128, 32, 1)),
        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(128, 32, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((1, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((1, 2)),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((1, 2)),
        keras.layers.Reshape((32, 256)),
        keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
        keras.layers.Dense(80, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(80)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


# def show_first_image():
#     img = preprocess(cv2.imread(FilePaths.fnInfer, cv2.IMREAD_GRAYSCALE), (128, 32))
#     # img = preprocess(cv2.imread('../data/words/a01/a01-000u/a01-000u-01-02.png', cv2.IMREAD_GRAYSCALE), (128, 32))
#     plt.figure()
#     plt.imshow(img, cmap=plt.cm.binary)
#     plt.colorbar()
#     plt.grid(False)
#     plt.show()

# show_first_image()
# sys.exit(0)

batch_size = 32
img_size = (128, 32)
max_text_len = 32

model = build_model()
# model.summary()


loader = WordsLoaderDataset(FilePaths.fnTrain, batch_size, img_size, max_text_len)
train_ds = loader.get_train_dataset(img_size)
val_ds = loader.get_validation_dataset(img_size)
char_list = train_ds.class_names
print('-----------Char list-----------------', train_ds.class_names)


checkpoint_dir = path.dirname(FilePaths.fnCheckpoint)
lastest_cp = tf.train.latest_checkpoint(checkpoint_dir)

if lastest_cp is not None:
    model.load_weights(lastest_cp)
else:
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=FilePaths.fnCheckpoint,
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=5,
        callbacks=[cp_callback]
    )


probability_model = keras.Sequential([
    model,
    keras.layers.Softmax()
])
# probability_model.summary()

img = preprocess(cv2.imread(FilePaths.fnInfer, cv2.IMREAD_GRAYSCALE), img_size)
predictions = probability_model.predict(np.array([img]))
prediction = predictions[0]

word_predicted = ''
for i in range(len(prediction)):
    step = prediction[i]
    word_predicted += char_list[np.argmax(step)]
word_predicted = word_predicted.strip()

plt.figure()
rotate_img = ndimage.rotate(img, 90)
plt.imshow(rotate_img, origin='lower', cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.xlabel(word_predicted)
plt.show()