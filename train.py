from model_handler import ModelHandler
from get_data import get_data

def train():
    '''train model and return that'''
    images, labels = get_data()
    model_handler = ModelHandler(images, labels)
    model_handler.set_models()
    model_handler.train(50)
    return model_handler
