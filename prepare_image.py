'''class for keeping images for more convenient usage'''
import cv2
from segmentation import segmantate, border


class PrepareImage:
    '''class for transforming image depended on needed process'''
    def __init__(self, image, path=''):
        if path != '':
            self.id = path[-10:-4]
        else:
            self.id = ''    
        self.path = path
        self.image = image
        
        self.segmanated = self.to_segmentate()
        self.border = self.get_border()
        self.centr = self.get_central_part()

    def to_segmentate(self, step=15):
        '''return segmentated image'''
        return segmantate(self.image, step)

    def compact(self, percent=50):
        '''return smaller image'''
        width = int(self.image.shape[1] * percent / 100)
        height = int(self.image.shape[0] * percent / 100)
        dim = (width, height)
        resized = cv2.resize(self.image, dim, interpolation = cv2.INTER_AREA)
        return resized

    def get_border(self):
        '''return image that contain only border of self'''
        img = border(self.image)
        width = int(img.shape[1] * 50 / 100)
        height = int(img.shape[0] * 50 / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized


    def get_central_part(self):
        '''return central part of image''' 
        width = int(self.image.shape[1])
        height = int(self.image.shape[0])
        cropped = self.image[int(width/4):int(3*width/4),
                    int(height/4):int(3*height/4)]
        return cropped  

    def rotate(self, angle):
        '''return rotated image'''
        (h, w, d) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    def set_img(self, clas):
        '''return different looks of image considering to clas'''
        if clas == 'Class1':
            return self.segmanated
        if clas in ['Class2', 'Class7']:
            return self.border
        if clas in ['Class3', 'Class9']:
            return self.centr
        if clas in ['Class4', 'Class6', 'Class8', 'Class10', 'Class11']:
            return self.image
        if clas == 'Class5':
            return segmantate(self.centr)
