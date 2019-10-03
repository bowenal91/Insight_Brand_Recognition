import time
import cv2
import gluoncv as gcv
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from ffmpy import FFmpeg

def run_detector(videofile):

    threshold = 0.75

    classes = ['Visa','Powerade','Hyundai','Coke','Adidas']
    stats = np.zeros(len(classes))

    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes = classes, pretrained_base=False)
    net.load_parameters('logos.params')
#net.collect_params().reset_ctx([mx.gpu(0)])
    cap = cv2.VideoCapture(videofile)
#out = cv2.VideoWriter('detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
    axes = None
    NUM_FRAMES = 6000
    inc = 1.0/float(NUM_FRAMES)
    img_array = []
    running = np.zeros((NUM_FRAMES,len(classes)))
    for i in range(NUM_FRAMES):
        try:
            ret,frame = cap.read()
        except:
            break
        frame = mx.nd.array(frame).astype('uint8')
        rgb_nd,frame = gcv.data.transforms.presets.ssd.transform_test(frame,short=512,max_size=700)
        #rgb_nd = rgb_nd.as_in_context(mx.gpu(0))
        class_IDs,scores,bounding_boxes = net(rgb_nd)
        for j in range(len(scores[0])):
            if scores[0][j] >= threshold:
                thisClass = int(class_IDs[0][j].asnumpy())
                bb = bounding_boxes[0][j].asnumpy()
                A = (bb[2]-bb[0])*(bb[3]-bb[1])
                stats[thisClass] += A*inc
                running[i][thisClass] += A*inc
            else:
                break
        img = gcv.utils.viz.cv_plot_bbox(frame,bounding_boxes[0],scores[0],class_IDs[0],class_names=net.classes,thresh=threshold)

        height,width,layers = img.shape
        size = (width,height)
        img_array.append(img)
        print(i)
        #out.write(img)
        #cv2.imshow('image',img)
        #gcv.utils.viz.cv_plot_image(img)
        #cv2.waitKey(1)

    for i in range(1,NUM_FRAMES):
        running[i] = running[i] + running[i-1]

    x = np.arange(len(classes))
    y = stats
    plt.figure()
#plt.subplot(211)
    plt.bar(x,y,edgecolor='k',linewidth=2,color=['black','red','green','blue','yellow'])
    plt.tick_params(axis='both',which='major',labelsize=12)
    plt.xticks(x,classes,fontsize=8)
    plt.ylabel("Appearances",fontsize=14)
    plt.savefig("app/static/Plot1.png")

    x2 = np.array(range(NUM_FRAMES))/45.0
    plt.figure()
#plt.subplot(212)
    cs = ['black','red','green','blue','yellow']
    for i in range(len(classes)):
        plt.plot(x2,running[:,i],color=cs[i],linewidth=2)

    plt.legend(classes)
    plt.xlim(0.0,float(NUM_FRAMES)/45.0)
    plt.tick_params(axis='both',which='major',labelsize=12)
    plt.xlabel("Time (seconds)",fontsize=14)
    plt.ylabel("Appearances")
    plt.savefig("app/static/Plot2.png")

    out = cv2.VideoWriter('app/static/detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),45,size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    ff = FFmpeg(inputs={'app/static/detected.avi':None}, outputs = {'app/static/final_video.mp4':'-an -vcodec libx264 -crf 23'})
    ff.run()

    print("Calculation complete")
    return

