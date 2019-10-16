import numpy as np
from PIL import Image
from PIL import ImageFilter
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img


def find_coeffs(pa,pb):
    """Determines coefficients to be used in nonlinear perpective transformation"""
    matrix = []
    for p1,p2 in zip(pa,pb):
        matrix.append([p1[0],p1[1],1,0,0,0,-p2[0]*p1[0],-p2[0]*p1[1]])
        matrix.append([0,0,0,p1[0],p1[1],1,-p2[1]*p1[0],-p2[1]*p1[1]])

    A = np.matrix(matrix,dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T*A)*A.T,B)
    return np.array(res).reshape(8)

#Class to manage images and prepare them for training

class Image_Handler:
    def __init__(self, filename, prop):
        self.bg_static = load_img(filename)
        self.bg = load_img(filename)
        self.bg = self.bg.resize((512,512))
        self.pix_bg = img_to_array(self.bg)

        self.prop = prop
        #self.logo_size = (int(prop*self.bg.size[0]),int(prop*self.bg.size[1]))
        self.logo_size = None
        self.logo = None
        self.logo_transformed = None
        self.label_list = []
        self.class_list = []
        self.logo_class = None

    def create_logo(self,filename,classID):
        """standardizes the logo to be superimposed"""
        self.logo = load_img(filename)
        a = self.prop*float(self.bg.size[1])
        b = float(self.logo.size[0])/float(self.logo.size[1])
        b = int(a*b)
        a = int(a)
        self.logo_size = (b,a)
        self.logo_class = classID
        #self.logo = self.logo.resize(self.logo_size)

    def transform_logo(self):
        #Perform a single random transformation on the logo
        #These parameters create bounds for the random transformation

        width = self.logo.size[0]
        height = self.logo.size[1]

        if np.random.uniform() < 0.7:

            #Perform perspective transformation
            #print("PERSPECTIVE")
            r = np.random.uniform()
            if r < 0.3:
                width_shift = width*np.random.uniform(-0.5,0.5)
                height_shift = 0
            else:
                width_shift = 0
                height_shift = height*np.random.uniform(-2.5,2.5)
            if height_shift < 0:
                h_origin = -height_shift
            else:
                h_origin = 0
            if width_shift < 0:
                w_origin = -width_shift
            else:
                w_origin = 0
            #print((width_shift,height_shift))
            r = np.random.uniform()
            coeffs = find_coeffs([(w_origin,h_origin), (width+w_origin,height_shift+h_origin), (width_shift+w_origin,height+h_origin), (width+width_shift+w_origin,height+height_shift+
                h_origin)],[(0,0), (width,0), (0,height), (width,height)])

            self.logo_transformed = self.logo.transform((width+abs(int(width_shift)),height+abs(int(height_shift))),Image.PERSPECTIVE,coeffs,Image.BICUBIC)

            """
            #perform affine transformation
            #print("AFFINE")
            m = np.random.uniform(-0.3,0.3)
            xshift = abs(m)*height
            new_height = height + int(round(xshift))
            #coeffs = (1,m,-xshift if m>0 else 0,0,1,0)
            coeffs = (1,0,0,0,1,0)

            self.logo_transformed = self.logo.transform((width,new_height),Image.AFFINE,coeffs,Image.BICUBIC)
            #print(self.logo_transformed.size)
            """

        else:
            #Perform perspective transformation
            #print("PERSPECTIVE")
            r = np.random.uniform()
            if r < 0.5:
                width_shift = width*np.random.uniform(0.0,0.2)
                height_shift = 0
            else:
                width_shift = 0
                height_shift = height*np.random.uniform(0.0,0.2)
            #print((width_shift,height_shift))
            r = np.random.uniform()
            if r < 0.5:
                coeffs = find_coeffs([(0,0), (width,height_shift), (width_shift,height), (width-width_shift,height-height_shift)],
                    [(0,0), (width,0), (0,height), (width,height)])
            else:
                coeffs = find_coeffs([(width_shift,height_shift), (width-width_shift,0), (0,height-height_shift), (width,height)],
                    [(0,0), (width,0), (0,height), (width,height)])

            self.logo_transformed = self.logo.transform((width,height),Image.PERSPECTIVE,coeffs,Image.BICUBIC)

        #Randomly rotate the resultant image
        #theta = np.random.uniform(-30.0,30.0)
        #self.logo_transformed = self.logo_transformed.rotate(theta,expand=True)

        #Randomly resize the image so that it takes up a different amount of space
        current_size = self.logo_transformed.size
        while True:
            rand = np.random.uniform(0.1,1.8)
            r1 = int(rand*self.logo_size[0])
            r2 = int(rand*self.logo_size[1])
            if r1 < self.bg.size[0] and r2 < self.bg.size[1]:
                break
        self.logo_transformed = self.logo_transformed.resize((r1,r2))


    def add_logo(self):
        #Picks a random location to add on the logo image
        #If the logo pixel at a given location is pure black, then we only use the background pixel value at that point
        bg_px = img_to_array(self.bg)
        logo_px = img_to_array(self.logo_transformed)
        #print(logo_px.shape)
        #print(self.logo_transformed.size)
        #Generate a random pixel location to add it in
        xlim = self.bg.size[0] - self.logo_transformed.size[0]
        ylim = self.bg.size[1] - self.logo_transformed.size[1]
        r_x = np.random.randint(xlim)
        r_y = np.random.randint(ylim)
        #print(r_x,r_y)
        r = np.random.randint(160,255)
        g = np.random.randint(160,255)
        b = np.random.randint(160,255)
        for j in range(self.logo_transformed.size[0]-1):
            for i in range(self.logo_transformed.size[1]-1):
                #print(i,j)
                if logo_px[i][j][0] > 10 or logo_px[i][j][1] > 10 or logo_px[i][j][2] > 10:
                    #bg_px[r_y+i,r_x+j,:] = logo_px[i,j,:]
                    bg_px[r_y+i,r_x+j,:] = (r,g,b)
                    #bg_px[r_y+i,r_x+j,:] = (255,255,255)

        self.bg = array_to_img(bg_px)

        #Generate a label and add it to the list
        img_x = float(self.bg.size[0])
        img_y = float(self.bg.size[1])

        label = [r_x, r_y, self.logo_transformed.size[0]+r_x, self.logo_transformed.size[1]+r_y]
        if self.logo_class != -1:
            self.label_list.append(label)
            self.class_list.append(self.logo_class)

    #def print_label(self,filename):
        #N = len(self.label_list)
        #with open(filename,"w") as f:
            #for l in self.label_list:
                #s = "{}\t{}\t{}\t{}\t{}\n".format(l[0],l[1],l[2],l[3],l[4])
                #f.write(s)

