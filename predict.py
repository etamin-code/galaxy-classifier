from numpy import array, argmax
import pathlib
import csv
from cut_and_scale_img import process_image
from train import train
from prepare_image import PrepareImage

def predict(images=[], PATH=None):
    '''path - path of the directory with images'''
    if images == [] and PATH is None:
        raise TypeError('images and PATH cant be both empty')
    if PATH is not None:
        images = []
        images_dir = pathlib.Path(PATH).iterdir()
        for path in images_dir:
            path = '\\'.join(str(path).split('\\'))
            images.append(PrepareImage(process_image(path), path))
        images = array(images)    
    model_handler = train()
    predictions = model_handler.predict(images, True)
    results = []
    for prediction in predictions:
        image = {'GalaxyID':prediction['id']}
        for key in list(prediction.keys())[1:]:
            image[key] = argmax(prediction[key]) / 20
        results.append(image)    
    with open('predictions.csv', "a", newline="") as file:
        columns = ["GalaxyID", 
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
        to_write = csv.DictWriter(file, fieldnames=columns)
        to_write.writeheader()     
        to_write.writerows(results)

    return predictions      
