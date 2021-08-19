'''process image to grayscaled, cutted, scaled image'''
import cv2

def read_path(path):
    '''return image object'''
    image = cv2.imread(path)
    return image

def cut_image(image):
    '''return cutted image, which is the central
    part of the image'''
    width = int(image.shape[1])
    height = int(image.shape[0])
    cropped = image[int(width/4):int(3*width/4),
                    int(height/4):int(3*height/4)]
    return cropped

def scale_image(image):
    '''scale image on 50 %'''
    width = int(image.shape[1] * 50 / 100)
    height = int(image.shape[0] * 50 / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def in_grey(image):
    '''return grayscaled image'''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def process_image(path):
    '''run procesed image'''
    return in_grey(scale_image(cut_image(read_path(path))))

def viewImage(image, name_of_window):
    '''show image'''
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = r"D:\programing\second_semester\home_work\galaxy_classifier\images\img7.png"
    img = process_image(path)
    viewImage(process_image(path), 'after')
    viewImage(cv2.imread(path) ,'before')
    # img = read_path(path)
    # viewImage(img, 'image')
    # img = cut_image(img)
    # viewImage(img, 'cutted')
    # img = scale_image(img)
    # viewImage(img)
    # img = in_grey(img)
    # viewImage(img, 'processed')
