from handlers import *
import glob
import sys

def generate_training_image(img_data,n_logo):
    for i in range(n_logo):
        img_data.transform_logo()
        img_data.add_logo()
    return img_data


if __name__ == "__main__":
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    label_dir = sys.argv[3]
    logo_file = sys.argv[4]
    N = 4000
    names = glob.glob(src_dir+"*")
    for i in range(N):
        print(i)
        j = np.random.randint(len(names))
        filename = names[j]
        im = Image_Handler(filename,0.2)
        im.create_logo(logo_file)
        n_logo = np.random.randint(1,13)
        generate_training_image(im,n_logo)
        im.print_label(label_dir+str(i)+".txt")
        im.bg.save(dest_dir+str(i)+".jpg")


