import time
import cv2
import gluoncv as gcv
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
from ffmpy import FFmpeg
import os

class Detector:
    def __init__(self,videofile,mode):
        self.threshold = 0.9

        #mode is 0 for detect only, 1 for tracking as well
        self.mode = mode
        self.classes = ['Visa','Powerade','Hyundai','Coke','Adidas']
        self.stats = np.zeros(len(self.classes))

        self.net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes = self.classes, pretrained_base=False)
        self.net.load_parameters('logos.params')
        #net.collect_params().reset_ctx([mx.gpu(0)])
        self.cap = cv2.VideoCapture(videofile)
        #out = cv2.VideoWriter('detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),15,size)
        self.NUM_FRAMES = 2000
        self.inc = 1.0/float(self.NUM_FRAMES)
        self.img_array = []
        self.out = None
        self.firstFrame = True
        self.running = np.zeros((self.NUM_FRAMES,len(self.classes)))
        self.scores = None
        self.class_IDs = None
        self.bounding_boxes = None
        self.cycle = 0
        if self.mode == 1:
            self.trackers = cv2.MultiTracker_create()
            self.bbs = []
        return

    def detect_frame(self):
        try:
            ret,frame = self.cap.read()
        except:
            pass
        frame = mx.nd.array(frame).astype('uint8')
        rgb_nd,frame = gcv.data.transforms.presets.ssd.transform_test(frame,short=512,max_size=700)
        #rgb_nd = rgb_nd.as_in_context(mx.gpu(0))
        if self.mode == 1:
            self.trackers = cv2.MultiTracker_create()
            self.bbs = []

        self.class_IDs,self.scores,self.bounding_boxes = self.net(rgb_nd)
        for j in range(len(self.scores[0])):
            if self.scores[0][j] >= self.threshold:
                thisClass = int(self.class_IDs[0][j].asnumpy())
                bb = self.bounding_boxes[0][j].asnumpy()
                A = (bb[2]-bb[0])*(bb[3]-bb[1])
                self.stats[thisClass] += A*self.inc
                self.running[self.cycle][thisClass] += A*self.inc
                if self.mode == 1:
                    tracker = cv2.TrackerCSRT.create()
                    tracker_bb = (bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1])
                    self.trackers.add(tracker,frame,tracker_bb)

            else:
                break
        img = gcv.utils.viz.cv_plot_bbox(frame,self.bounding_boxes[0],self.scores[0],self.class_IDs[0],class_names=self.net.classes,thresh=self.threshold)

        height,width,layers = img.shape
        self.size = (width,height)
        return img

    def track_frame(self):
        #Update the tracker to account for camera movement. Always assume mode=1
        try:
            ret,old_frame = self.cap.read()
        except:
            pass
        frame = cv2.resize(old_frame,self.size)
        (success,bbs) = self.trackers.update(frame)

        #Convert frame and bounding box to mx array form
        old_frame = mx.nd.array(frame).astype('uint8')
        rgb_nd,old_frame = gcv.data.transforms.presets.ssd.transform_test(old_frame,short=512,max_size=700)
        for j in range(len(self.scores[0])):
            if self.scores[0][j] > self.threshold:
                thisClass = int(self.class_IDs[0][j].asnumpy())
                A = bbs[j][2]*bbs[j][3]
                self.stats[thisClass] += A*self.inc
                self.running[self.cycle][thisClass] += A*self.inc

                self.bounding_boxes[0][j][0] = bbs[j][0]
                self.bounding_boxes[0][j][1] = bbs[j][1]
                self.bounding_boxes[0][j][2] = bbs[j][0]+bbs[j][2]
                self.bounding_boxes[0][j][3] = bbs[j][1]+bbs[j][3]
            else:
                break

        img = gcv.utils.viz.cv_plot_bbox(old_frame,self.bounding_boxes[0],self.scores[0],self.class_IDs[0],class_names=self.net.classes,thresh=self.threshold)

        return img

    def run_detector(self):

        for self.cycle in range(self.NUM_FRAMES):
            print(self.cycle)
            if (self.cycle%20==0 or self.mode == 0):
                img = self.detect_frame()
            else:
                img = self.track_frame()
            if self.firstFrame:
                self.out = cv2.VideoWriter('flaskapp/static/detected.avi',cv2.VideoWriter_fourcc(*'DIVX'),45,self.size)
                self.firstFrame = False
            self.out.write(img)
        for i in range(1,self.NUM_FRAMES):
            self.running[i] = self.running[i] + self.running[i-1]


        return

    def generate_plots(self):
        x = np.arange(len(self.classes))
        y = self.stats
        plt.figure()
#plt.subplot(211)
        plt.bar(x,y,edgecolor='k',linewidth=2,color=['black','red','green','blue','yellow'])
        plt.tick_params(axis='both',which='major',labelsize=12)
        plt.xticks(x,self.classes,fontsize=8)
        plt.ylabel("Appearances",fontsize=14)
        plt.savefig("flaskapp/static/Cumulative_Bar.png")

        x2 = np.array(range(self.NUM_FRAMES))/45.0
        plt.figure()
#plt.subplot(212)
        cs = ['black','red','green','blue','yellow']
        for i in range(len(self.classes)):
            plt.plot(x2,self.running[:,i],color=cs[i],linewidth=2)

        plt.legend(self.classes)
        plt.xlim(0.0,float(self.NUM_FRAMES)/45.0)
        plt.tick_params(axis='both',which='major',labelsize=12)
        plt.xlabel("Time (seconds)",fontsize=14)
        plt.ylabel("Appearances")
        plt.savefig("flaskapp/static/Run_Chart.png")
        return

    def write_video(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        try:
            os.remove('flaskapp/static/final_video.mp4')
        except:
            pass
        ff = FFmpeg(inputs={'flaskapp/static/detected.avi':None}, outputs = {'flaskapp/static/final_video.mp4':'-an -vcodec libx264 -crf 23'})
        ff.run()
        return

