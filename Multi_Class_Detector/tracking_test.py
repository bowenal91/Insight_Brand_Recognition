import time
import cv2
import gluoncv as gcv
import mxnet as mx

detect_interval = 15
detect_counter = 1
classes = ['Visa','Powerade','Hyundai']
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes = classes, pretrained_base=False)
net.load_parameters('logos.params')
#net.collect_params().reset_ctx([mx.gpu(0)])
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
    if detect_counter%detect_interval==0:
        #Detect the objects and set up trackers for them
        detect_counter = 1
        frame = mx.nd.array(frame).astype('uint8')
        rgb_nd,frame = gcv.data.transforms.presets.ssd.transform_test(frame,short=512,max_size=700)
        #rgb_nd = rgb_nd.as_in_context(mx.gpu(0))
        class_IDs,scores,bounding_boxes = net(rgb_nd)

        img = gcv.utils.viz.cv_plot_bbox(frame,bounding_boxes[0],scores[0],class_IDs[0],class_names=net.classes,thresh=0.99)

        height,width,layers = img.shape
        size = (width,height)
        img_array.append(img)
        print(i)
        #out.write(img)
        #cv2.imshow('image',img)
        #gcv.utils.viz.cv_plot_image(img)
        #cv2.waitKey(1)

    else:
        #Update the trackers
        A = 1

    #Check for a scene change - automatic object detection


out = cv2.VideoWriter('detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),60,size)

for i in range(len(img_array)):
    out.write(img_array[i])
cap.release()
out.release()
cv2.destroyAllWindows()

