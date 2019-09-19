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
    logo_file = sys.argv[3]
    N = 200
    names = glob.glob(src_dir+"*")
    with open("labels.txt","w") as out:
        for i in range(N):
            print(i)
            j = np.random.randint(len(names))
            filename = names[j]
            im = Image_Handler(filename,0.3)
            im.create_logo(logo_file)
            n_logo = np.random.randint(1,4)
            generate_training_image(im,n_logo)

            #im.print_label(label_dir+str(i)+".txt")
            im.bg.save(dest_dir+str(i)+".jpg")
            out.write(dest_dir+str(i)+".jpg\t")
            for box in im.label_list:
                out.write("{},{},{},{}\t".format(box[0],box[1],box[2],box[3]))
            out.write("\n")

