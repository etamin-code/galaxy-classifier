'''class for models training'''


import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import csv
from get_data import get_data
from prepare_image import PrepareImage


class MultModel(keras.Sequential):
    def set_data(self, clas, images, labels):
        '''set train and test data'''
        self.train_images = np.array([image.set_img(clas) for image in images])
        # set train_images as list of processed images special for that clas
        self.test_images = np.array([image.set_img(clas) for image in images[len(images) * 9 // 10:]])
        self.train_labels = labels[clas]
        self.test_labels = labels[clas][len(labels[clas]) * 9 // 10:]
        self.clas = clas

    def set_test_images(self, images):
        '''images - iterable(PrepareImage)
        return processed images special for that clas
        return np.array'''
        test_images = np.array([image.set_img(self.clas) for image in images])
        return test_images

    def fit(self, epochs=5):
        '''create and train models of the clas'''
        SIZE = len(self.train_images[0])
        NEYRONS = SIZE ** 2 // 6
        # number of neyrons which are used for the training
        # a little less then number of all pixels, to make process faster
        models = []
        num_of_models = len(self.train_labels[0])
        for i in range(1, num_of_models + 1):
            train_labels = np.array([el[i-1] for el in self.train_labels])
            model = keras.Sequential([
                        keras.layers.Flatten(input_shape=(SIZE, SIZE)),
                        keras.layers.Dense(NEYRONS, activation=tf.nn.relu),
                        keras.layers.Dense(21, activation=tf.nn.softmax)
                    ])
            # create model
            model.compile(optimizer=tf.keras.optimizers.Adam(), 
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            print('{}.{}'.format(self.clas, i))
            history = model.fit(self.train_images, train_labels, epochs=epochs)
            with open('history//history{}_{}.pickle'.format(self.clas, i), 'wb') as f:
                pickle.dump(history.history, f)
            # save history
            models.append(model)
        self.models = models

    def predict(self, images):
        '''return predicted characteristic for the images
            images - iterable(PrepareImage)
            return - list(list)'''
        if images is None:
            images = self.test_images
        else:
            images = self.set_test_images(images)    
        dict_of_predictions = dict()
        for num, model in enumerate(self.models):
            dict_of_predictions[num] = model.predict(images)
        predictions = []
        for i in range(len(list(dict_of_predictions.values())[0])):
            item = [clas[i] for clas in dict_of_predictions.values()]
            predictions.append(item)
        return predictions    
    
    def evaluate(self):
        '''return the accuracy of the model'''
        test_loss, test_acc = 0, 0
        num_of_models = len(self.test_labels[0])
        for i in range(1, num_of_models + 1):
            model = self.models[i-1]
            test_labels = np.array([el[i-1] for el in self.test_labels])
            loss, acc = model.evaluate(self.test_images, test_labels)
            test_loss += loss
            test_acc += acc
        test_loss, test_acc = test_loss / 3, test_acc / 3
        return test_loss, test_acc

class ModelHandler:
    '''class for keeping all models'''
    def __init__(self, images, labels):
        self.classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5',
         'Class6', 'Class7', 'Class8', 'Class9', 'Class10', 'Class11']
        self.sub_classes = [
            'Class1.1', 'Class1.2', 'Class1.3',
            'Class2.1', 'Class2.2',
            'Class3.1', 'Class3.2', 
            'Class4.1', 'Class4.2', 
            'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 
            'Class6.1', 'Class6.2', 
            'Class7.1', 'Class7.2', 'Class7.3', 
            'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 
            'Class9.1', 'Class9.2', 'Class9.3', 
            'Class10.1', 'Class10.2', 'Class10.3', 
            'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6'
            ]
        self.images = images
        self.labels = labels
        self.models = self.set_models()
        
    def train(self, epochs=5):
        '''train model'''
        for num, model in enumerate(self.models):
            clas = self.classes[num]
            model.set_data(clas, self.images, self.labels)
            model.fit(epochs)

    def evaluate(self):
        '''return accuracy of modelation'''
        test_loss, test_acc = 0, 0
        for model in self.models:
            loss, acc = model.evaluate()
            test_loss += loss
            test_acc += acc
        test_loss, test_acc = test_loss / 11, test_acc / 11
        return (test_loss, test_acc)

    def predict(self, images=None, show=False):
        '''return expected characteristic of the images
        return - list(dict)'''
        dict_of_classes = dict()
        # {'classi.j': [img1pred, img2pred...]}
        for model in self.models:
            dict_of_classes[model.clas] = model.predict(images)
        predictions = []
        # [{id: img_id, 'Class1.1': pred1.1, 'Class1.2': pred2...}...]
        for i in range(len(list(dict_of_classes.values())[0])):
            img = {'id': images[i].id}
            j = 0
            for clas_pred, clas_name in zip(dict_of_classes.values(), self.sub_classes):
                for sub_clas_pred in clas_pred[i]:
                    img[self.sub_classes[j]] = sub_clas_pred
                    j += 1
            predictions.append(img)
        str_predictions = ''
        for img in predictions:
            str_predictions += '\nimage {}'.format(img['id'])
            for sub_clas in self.sub_classes:
                str_predictions += '\n{}: {}'.format(sub_clas,
                                            np.argmax(predictions[i][sub_clas]) / 20)
        if show:
            print(str_predictions)                                    
        return predictions     

    def set_models(self):
        '''set models of self
        models - MultModel of every from the 11 classes
        of characteristics
        return list(MultModel)'''
        models = []
        for clas in self.classes:
            model = MultModel()
            model.set_data(clas, self.images, self.labels)    
            models.append(model)
        return models
