import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
import sys
from gluoncv.utils import download, viz

dataset = gcv.data.RecordFileDetection('validation.rec')
classes = ['Visa','Powerade','Hyundai','Coke','Adidas']  # only one foreground class here

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,
    pretrained_base=False, transfer='voc')

def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

train_data = get_dataloader(net, dataset, 512, 16, 0)

try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('logos.params')
threshold = 0.8
IoU_avg = 0.0
counter = 0.0
for i in range(len(dataset)):
    print(i)
    image,label = dataset[i]
    frame = mx.nd.array(image).astype('uint8')
    rgb_nd,frame = gcv.data.transforms.presets.ssd.transform_test(frame,short=512,max_size=700)

    gtboxes =label[:,:4]
    gtlabels = label[:,4:5]
    classIDs,scores,bbs = net(rgb_nd)
    bbs = bbs[0].asnumpy()
    scores = scores[0].asnumpy()
    classIDs = classIDs[0].asnumpy()
    for j in range(len(scores)):
        if scores[j] >= threshold:
            bb = bbs[j]
            ID = classIDs[j]
            maxScore = 0.0
            maxID = -1
            for k in range(len(gtboxes)):
                #Calculate IoU for the two boxes
                if gtlabels[k] != ID:
                    continue
                gtbb = gtboxes[k]
                #print(gtbb)
                xmin = max(bb[0],gtbb[0])
                ymin = max(bb[1],gtbb[1])
                xmax = min(bb[2],gtbb[2])
                ymax = min(bb[3],gtbb[3])
                if xmin > xmax or ymin > ymax:
                    intersect = 0.0
                else:
                    intersect = (xmax-xmin)*(ymax-ymin)
                union = (bb[2]-bb[0])*(bb[3]-bb[1]) + (gtbb[2]-gtbb[0])*(gtbb[3]-gtbb[1]) - intersect
                IOU = intersect/union
                if IOU > maxScore:
                    maxScore = IOU
                    maxID = k
            # Get the ground truth box with the best match and delete it from the array

            if maxID != -1:
                IoU_avg += maxScore
                counter += 1.0
                #print(gtboxes)
                gtboxes = np.delete(gtboxes,maxID,0)
                #print(gtboxes)
                gtlabels = np.delete(gtlabels,maxID,0)
            else:
                counter += 1.0

        else:
            break


IoU_avg /= counter
print(IoU_avg)


