from PIL import Image
import os
import glob

names = glob.glob("*.jpg")
for filename in names:
    delete = False
    try:
        im = Image.open(filename)
        if im is None:
            delete = True
    except:
        delete = True

    if delete:
        print("Deleting {}".format(filename))
        os.remove(filename)

