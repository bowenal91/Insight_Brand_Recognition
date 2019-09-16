from PIL import Image

class Image_Handler:
    def __init__(self, filename, filename2):
        self.bg = Image.open(filename)
        self.logo = Image.open(filename2)
        self.pix = self.im.load()
        self.size = self.im.size
    def distort:
        #Return a random distortion of the image
