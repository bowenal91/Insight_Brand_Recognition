import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img



#Class to manage images and prepare them for training

class Image_Handler:
    def __init__(self, filename, bg_size, logo_size):
        self.bg_static = load_img(filename)
        self.bg = load_img(filename)
        self.bg = self.bg.resize(bg_size)
        self.pix_bg = img_to_array(self.bg)
        self.logo_size = logo_size
        self.logo = None
        self.logo_transformed = None

    def add_logo(self,filename):
        self.logo = load_img(filename)
        self.logo = self.logo.resize(self.logo_size)

    def transform_logo(self):
        #Perform a single random transformation on the logo
        #These parameters create bounds for the random transformation

    def add_logo(self):
        #Picks a random location to add on the logo image
        #If the logo pixel at a given location is pure black, then we only use the background pixel value at that point

