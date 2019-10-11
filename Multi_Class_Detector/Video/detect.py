import time
import cv2
import gluoncv as gcv
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from ffmpy import FFmpeg
import os

class Detector:
    def __init__(self,videofile):
        self.threshold = 0.8
        self.threshold = 0.8

        self.classes = ['Visa','Powerade','Hyundai','Coke','Adidas']
        self.stats = np.zeros(len(classes))

        self.net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes = classes, pretrained_base=False)
        self.net.load_parameters('logos.params')
        #net.collect_params().reset_ctx([mx.gpu(0)])
        self.cap = cv2.VideoCapture(videofile)
        #out = cv2.VideoWriter('detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
        self.NUM_FRAMES = 10
        self.inc = 1.0/float(NUM_FRAMES)
        self.img_array = []
        self.running = np.zeros((NUM_FRAMES,len(classes)))
        return

    def detect_frame(self):
        try:
            ret,frame = cap.read()
        except:
            break
        frame = mx.nd.array(frame).astype('uint8')
        rgb_nd,frame = gcv.data.transforms.presets.ssd.transform_test(frame,short=512,max_size=700)
        #rgb_nd = rgb_nd.as_in_context(mx.gpu(0))
        class_IDs,scores,bounding_boxes = self.net(rgb_nd)
        for j in range(len(scores[0])):
            if scores[0][j] >= self.threshold:
                thisClass = int(class_IDs[0][j].asnumpy())
                bb = bounding_boxes[0][j].asnumpy()
                A = (bb[2]-bb[0])*(bb[3]-bb[1])
                self.stats[thisClass] += A*inc
                self.running[i][thisClass] += A*inc
            else:
                break
        img = gcv.utils.viz.cv_plot_bbox(frame,bounding_boxes[0],scores[0],class_IDs[0],class_names=net.classes,thresh=threshold)

        height,width,layers = img.shape
        self.size = (width,height)
        return


    def run_detector(self):
        for i in range(NUM_FRAMES):
            detect_frame()
        for i in range(1,NUM_FRAMES):
            running[i] = running[i] + running[i-1]


        self.out = cv2.VideoWriter('flaskapp/static/detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),45,self.size)
        return

    def generate_plots(self):
        x = np.arange(len(classes))
        y = stats
        plt.figure()
#plt.subplot(211)
        plt.bar(x,y,edgecolor='k',linewidth=2,color=['black','red','green','blue','yellow'])
        plt.tick_params(axis='both',which='major',labelsize=12)
        plt.xticks(x,classes,fontsize=8)
        plt.ylabel("Appearances",fontsize=14)
        plt.savefig("flaskapp/static/Cumulative_Bar.png")

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
        plt.savefig("flaskapp/static/Run_Chart.png")
        return

    def write_video(self):
        for i in range(len(img_array)):
            out.write(img_array[i])
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        try:
            os.remove('flaskapp/static/final_video.mp4')
        except:
            pass
        ff = FFmpeg(inputs={'flaskapp/static/detected.avi':None}, outputs = {'flaskapp/static/final_video.mp4':'-an -vcodec libx264 -crf 23'})
        ff.run()
        return

