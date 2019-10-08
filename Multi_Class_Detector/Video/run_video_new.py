import time
import cv2
import gluoncv as gcv
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt


classes = ['Visa','Powerade','Hyundai','Coke','Adidas']
stats = np.zeros(len(classes))

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes = classes, pretrained_base=False)
net.load_parameters('logos.params')
net.collect_params().reset_ctx([mx.gpu(0)])
cap = cv2.VideoCapture('soccer.mp4')
#out = cv2.VideoWriter('detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
axes = None
NUM_FRAMES = 6000
img_array = []
for i in range(NUM_FRAMES):
    try:
        ret,frame = cap.read()
    except:
        break
    frame = mx.nd.array(frame).astype('uint8')
    rgb_nd,frame = gcv.data.transforms.presets.ssd.transform_test(frame,short=512,max_size=700)
    rgb_nd = rgb_nd.as_in_context(mx.gpu(0))
    class_IDs,scores,bounding_boxes = net(rgb_nd)
    for j in range(len(scores[0])):
        if scores[0][j] >= 0.8:
            thisClass = int(class_IDs[0][j].asnumpy())
            stats[thisClass] += 1.0
        else:
            break
    img = gcv.utils.viz.cv_plot_bbox(frame,bounding_boxes[0],scores[0],class_IDs[0],class_names=net.classes,thresh=0.8)

    height,width,layers = img.shape
    size = (width,height)
    img_array.append(img)
    print(i)
    #out.write(img)
    #cv2.imshow('image',img)
    #gcv.utils.viz.cv_plot_image(img)
    #cv2.waitKey(1)

x = np.arange(len(classes))
y = stats
plt.bar(x,y,edgecolor='k',linewidth=2,color=['black','red','green','blue','yellow'])
plt.tick_params(axis='both',which='major',labelsize=12)
plt.xticks(x,classes,fontsize=8)
plt.ylabel("Appearances",fontsize=14)
plt.savefig("Plot.png")

out = cv2.VideoWriter('detected.mp4',cv2.VideoWriter_fourcc(*'H264'),45,size)

for i in range(len(img_array)):
    out.write(img_array[i])
cap.release()
out.release()
cv2.destroyAllWindows()

