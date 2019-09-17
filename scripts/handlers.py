import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img


def find_ceoffs(pa,pb):
    matrix = []
    for p1,p2 in zip(pa,pb):
        matrix.append([p1[0],p1[2],1,0,0,0,-p2[0]*p1[0],-p2[0]*p1[1]])
        matrix.append([0,0,0,p1[0],p1[1],1,-p2[1]*p1[0],-p2[1]*p1[1]])

    A = np.matrix(matrix,dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T*A)*A.T,B)
    return np.array(res).reshape(8)

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
        width = logo_size[0]
        height = logo_size[1]

        if np.random.uniform() < 1.0:
            #perform affine transformation

            m = -0.5
            xshift = abs(m)*width
            new_width = width + int(round(xshift))
            coeffs = (1,m,-xshift if m>0 else 0,0,1,0)

            self.logo_transformed = self.logo.tranform((width,height),Image.AFFINE,coeffs,Image.BICUBIC)

        else:
            #Perform perspective transformation
            r = np.random.uniform()
            if r < 0.5:
                width_shift = width*np.random.uniform(0.8,1.0)
                height_shift = 0
            else:
                width_shift = 0
                height_shift = height*np.random.uniform(0.8,1.0)
            coeffs = find_coeffs(
                    [(0,0), (width,0), (0,height), (width,height)],
                    [(width_shift,height_shift), (width-width_shift,height_shift), (width_shift,height-height_shift), (width-width_shift,height-height_shift)])

            self.logo_transformed = self.logo.transform((width,height),Image.PERSPECTIVE,coeffs,Image.BICUBIC)

    def add_logo(self):
        #Picks a random location to add on the logo image
        #If the logo pixel at a given location is pure black, then we only use the background pixel value at that point
        bg_px = img_to_array(self.bg)
        logo_px = img_to_array(self.logo)



