import os
import numpy as np
import cv2
import math
from SamplePreprocessor import preprocess

import tensorflow as tf


class Sample:
    "sample from the dataset"
    def __init__(self, img=None, label='', gtText='', filename=''):
        self.img = img
        self.label = label
        self.gtText = gtText
        self.filename = filename

class WordsLoaderDataset(object):
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

    def __init__(self, filePath, batchSize, imgSize, maxTextLen, validation_percentage=0.1):
        "loader for dataset at given location, preprocess images and text according to parameters"

        assert filePath[-1]=='/'

        self.filePath = filePath
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.maxTextLen = maxTextLen
        self.samples = []

        chars = set()
        bad_samples = []
        bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']

        with open(self.getFilename()) as f:
            for line in f:
                # ignore comment line
                if not line or line[0]=='#':
                    continue
                
                lineSplit = line.strip().split(' ')
                assert len(lineSplit) >= 9
                
                # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
                fileNameSplit = lineSplit[0].split('-')
                fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

                # GT text are columns starting at 9
                gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
                chars = chars.union(set(list(gtText)))

                # check if image is not empty
                if not os.path.getsize(fileName):
                    bad_samples.append(lineSplit[0] + '.png')
                    continue

                # put sample into list
                self.samples.append(Sample(gtText=gtText, filename=fileName))

        # some images in the IAM dataset are known to be damaged, don't show warning for them
        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        self.charList = sorted(list(chars))
        self.set_count = len(self.samples)
        self.offset = self.set_count - math.floor(self.set_count * validation_percentage)

        print(f'Dataset count = {self.set_count}, Dataset offset = {self.offset}')


    def getFilename(self):
        return self.filePath + 'words.txt'

    def getNextSample(self, index, dataAugmentation=False):
        sample = self.samples[index]
        # create sample img
        img = preprocess(cv2.imread(sample.filename, cv2.IMREAD_GRAYSCALE), self.imgSize, dataAugmentation)
        
        label = []
        for i in range(self.maxTextLen):
            if i < len(sample.gtText):
                label.append(self.charList.index(sample.gtText[i]))
            else:
                label.append(self.charList.index(' '))

        sample.img = img
        sample.label = label
        return sample



    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input 
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text

    def _generator(self, start, end, dataAugmentation):
        for i in range(start, end):
            sample = self.getNextSample(i, dataAugmentation=dataAugmentation)
            if sample is None:
                continue
            t = (sample.img, sample.label)
            yield t

    def get_train_dataset(self, output_shape):
        dataset = tf.data.Dataset.from_generator(
            self._generator, 
            output_types=(tf.dtypes.float32, tf.dtypes.int32),
            output_shapes=(output_shape, (None,)),
            args=(0, self.offset, True)
        )
        dataset = dataset.batch(self.batchSize)
        dataset.class_names = self.charList
        return dataset


    def get_validation_dataset(self, output_shape):
        dataset = tf.data.Dataset.from_generator(
            self._generator, 
            output_types=(tf.dtypes.float32, tf.dtypes.int32),
            output_shapes=(output_shape, (None,)),
            args=(self.offset, self.set_count, False)
        )
        dataset = dataset.batch(self.batchSize)
        dataset.class_names = self.charList
        return dataset

