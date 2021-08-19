'''return numpy arrays with images and data'''
import csv
import pathlib
from numpy import array
from cut_and_scale_img import process_image 
from prepare_image import PrepareImage

def get_data(): 
    '''return lists of the images and labels'''   
    TRAIN_IMAGES_PATH = r'D:\programing\second_semester\home_work\galaxy-zoo-the-galaxy-challenge\images_training_rev1\images_training_rev1'
    TRAIN_CHARACTERISTIC_PATH = r'D:\programing\second_semester\home_work\galaxy-zoo-the-galaxy-challenge\training_solutions_rev1\training_solutions_rev1.csv'
    
    NUM_OF_TRAINS = 5000
    # number of used images

    train_images = []
    images_dir = pathlib.Path(TRAIN_IMAGES_PATH).iterdir()
    i = 0

    for path in images_dir:
        path = '\\'.join(str(path).split('\\'))
        train_images.append(PrepareImage(process_image(path), path))
        i += 1
        if i == NUM_OF_TRAINS:
            break

    train_labels = {'Class1': [], 'Class2': [], 'Class3': [], 'Class4': [], 
                   'Class5': [], 'Class6': [], 'Class7': [], 'Class8': [], 
                   'Class9': [], 'Class10': [], 'Class11': []
                   }

    with open(TRAIN_CHARACTERISTIC_PATH) as f:
        characteristics = csv.DictReader(f, delimiter=',')
        i = 0
        for galaxy in characteristics:
            train_labels['Class1'].append([int(float(galaxy['Class1.1']) * 20),
                                          int(float(galaxy['Class1.2']) * 20),
                                          int(float(galaxy['Class1.3']) * 20)])
            
            train_labels['Class2'].append([round(float(galaxy['Class2.1']) * 20, 0),
                                          round(float(galaxy['Class2.2']) * 20, 0)])
            
            train_labels['Class3'].append([round(float(galaxy['Class3.1']) * 20, 0),
                                          round(float(galaxy['Class3.2']) * 20, 0)])
            
            train_labels['Class4'].append([round(float(galaxy['Class4.1']) * 20, 0),
                                          round(float(galaxy['Class4.2']) * 20, 0)])
            
            train_labels['Class5'].append([round(float(galaxy['Class5.1']) * 20, 0),
                                          round(float(galaxy['Class5.2']) * 20, 0),
                                          round(float(galaxy['Class5.3']) * 20, 0),
                                          round(float(galaxy['Class5.4']) * 20, 0)])
            
            train_labels['Class6'].append([round(float(galaxy['Class6.1']) * 20, 0),
                                          round(float(galaxy['Class6.2']) * 20, 0)])
            
            train_labels['Class7'].append([round(float(galaxy['Class7.1']) * 20, 0),
                                          round(float(galaxy['Class7.2']) * 20, 0),
                                          round(float(galaxy['Class7.3']) * 20, 0)])
            
            train_labels['Class8'].append([round(float(galaxy['Class8.1']) * 20, 0),
                                          round(float(galaxy['Class8.2']) * 20, 0),
                                          round(float(galaxy['Class8.3']) * 20, 0),
                                          round(float(galaxy['Class8.4']) * 20, 0),
                                          round(float(galaxy['Class8.5']) * 20, 0),
                                          round(float(galaxy['Class8.6']) * 20, 0),
                                          round(float(galaxy['Class8.7']) * 20, 0)])
            
            train_labels['Class9'].append([round(float(galaxy['Class9.1']) * 20, 0),
                                          round(float(galaxy['Class9.2']) * 20, 0),
                                          round(float(galaxy['Class9.3']) * 20, 0)])
            
            train_labels['Class10'].append([round(float(galaxy['Class10.1']) * 20, 0),
                                           round(float(galaxy['Class10.2']) * 20, 0),
                                           round(float(galaxy['Class10.3']) * 20, 0)])   

            train_labels['Class11'].append([round(float(galaxy['Class11.1']) * 20, 0),
                                           round(float(galaxy['Class11.2']) * 20, 0),
                                           round(float(galaxy['Class11.3']) * 20, 0),
                                           round(float(galaxy['Class11.4']) * 20, 0),
                                           round(float(galaxy['Class11.5']) * 20, 0),
                                           round(float(galaxy['Class11.6']) * 20, 0)])                                                                                                                                                                                                      
            
            i += 1
            if i == NUM_OF_TRAINS:
                break
            
    train_labels = {key: array(train_labels[key]) for key in train_labels.keys()}

    return (array(train_images), train_labels)  

if __name__ == '__main__':
    get_data()
        