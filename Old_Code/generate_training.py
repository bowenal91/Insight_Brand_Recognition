from handlers import *
import glob
import sys
import mxnet as mx

def write_line(img_path, im_shape, boxes, ids, idx):
    h, w, c = im_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4
    B = 5
    C = w
    D = h
    # concat id and bboxes
    labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
    # normalized bboxes (recommanded)
    labels[:, (1, 3)] /= float(w)
    labels[:, (2, 4)] /= float(h)
    # flatten
    labels = labels.flatten().tolist()
    str_idx = [str(idx)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_labels = [str(x) for x in labels]
    str_path = [img_path]
    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
    return line

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
    with open("labels.lst","w") as out:
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
            img = mx.image.imread(dest_dir+str(i)+".jpg")
            #out.write(dest_dir+str(i)+".jpg\t")
            all_boxes = np.array(im.label_list)
            all_ids = np.array([0]*len(im.label_list))
            #print(all_boxes)
            line = write_line(dest_dir+str(i)+".jpg",img.shape,all_boxes,all_ids,i)

            out.write(line)
            #out.write("0\n")

