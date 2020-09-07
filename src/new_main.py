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


batch_size = 32
img_size = (128, 32)
max_text_len = 32


def ctc_loss(y_true, y_pred):
    input_length_samples = tf.math.count_nonzero(y_true, axis=1, keepdims=True)
    input_placeholder = tf.fill(tf.shape(input_length_samples), max_text_len)
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_placeholder, input_length_samples)


def ctc_decode(y_pred):
    input_length = tf.math.count_nonzero(tf.math.argmax(y_pred, axis=2), axis=1)
    return keras.backend.ctc_decode(
        y_pred,
        input_length,
        greedy=False,
        beam_width=max_text_len,
        top_paths=1
    )[0]


def build_model(is_train_model=True, print_summary=False):
    model = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 32)),
        keras.layers.Reshape((128, 32, 1)),
        keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(128, 32, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((1, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((1, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((1, 2)),
        keras.layers.Reshape((32, 256)),
        keras.layers.Dropout(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
        # keras.layers.Dense(80, activation='relu'),
        # keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
        keras.layers.TimeDistributed(keras.layers.Dense(80)),
        keras.layers.Softmax()
    ])

    if not is_train_model:
        model.add(keras.layers.Lambda(ctc_decode, output_shape=(None, None), name='CTCdecode'))

    model.compile(
        optimizer='rmsprop',
        loss=ctc_loss,
        metrics=['accuracy']
    )

    if print_summary:
        model.summary()

    return model




loader = WordsLoaderDataset(FilePaths.fnTrain, batch_size, img_size, max_text_len)
train_ds = loader.get_train_dataset(img_size)
val_ds = loader.get_validation_dataset(img_size)
char_list = train_ds.class_names
print('-----------Char list-----------------', train_ds.class_names)

model = build_model(print_summary=True)

checkpoint_dir = path.dirname(FilePaths.fnCheckpoint)
lastest_cp = tf.train.latest_checkpoint(checkpoint_dir)
reload_latest_cp = False

if lastest_cp is not None:
    model.load_weights(lastest_cp)
else:
    reload_latest_cp = True
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
        epochs=10,
        callbacks=[cp_callback]
    )

if reload_latest_cp:
    lastest_cp = tf.train.latest_checkpoint(checkpoint_dir)
probability_model = build_model(is_train_model=False)
probability_model.load_weights(lastest_cp)

imgs_to_predict = []
for infer in FilePaths.fnInfer:
    img = preprocess(cv2.imread(infer, cv2.IMREAD_GRAYSCALE), img_size)
    imgs_to_predict.append(img)
predictions = probability_model.predict(np.array(imgs_to_predict))[0]

words_predicted = []
for i in range(len(imgs_to_predict)):
    prediction = predictions[i]
    wp = ''
    for current_step_time in range(len(prediction)):
        char_index = prediction[current_step_time]
        if char_index >= 0 and char_index < len(char_list):
            wp += char_list[char_index]
    # wp = wp.strip()
    words_predicted.append(wp)
    

plt.figure()
for i in range(len(imgs_to_predict)):
    plt.subplot(len(imgs_to_predict), 1, i + 1)
    rotate_img = ndimage.rotate(imgs_to_predict[i], 90)
    plt.imshow(rotate_img, origin='lower', cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(words_predicted[i])
plt.show()