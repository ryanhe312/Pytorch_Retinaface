import os
import numpy as np
import matplotlib.pyplot as plt
from bbox import BBox2DList
from bbox.metrics import multi_iou_2d

def read_bbox_file(filepath,mode='predict'):

    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]

    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        box = [float(line[0]), float(line[1]), float(line[2])-float(line[0]), float(line[3])-float(line[1])]
        if mode == 'predict':
            box.append(float(line[4]))
        boxes.append(box)
    return np.array(boxes)

def image_eval(pred, gt, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    success = np.zeros(_pred.shape[0])
    predict = np.zeros(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = multi_iou_2d(BBox2DList(_pred[:, :4]),BBox2DList(_gt))

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            success[h] = 1
            predict[h] = max_idx
    return success==1, predict

def img_pr_info(thresh_num, pred_info, success, predict):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.array(pred_info[:, 4] >= thresh)
        if np.sum(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index & success
            pr_info[t, 0] = np.sum(r_index)
            pr_info[t, 1] = len(set(predict[r_index]))
    return pr_info

def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve

def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

if __name__ == '__main__':

    predict_path = './test/Predicts'
    gt_path = './test/Labels'
    thresh_num = 1000
    iou_thresh = 0.5

    count_face = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for file in os.listdir(gt_path):
        #print(file)
        pred_info  = read_bbox_file(os.path.join(predict_path,file))
        gt_boxes   = read_bbox_file(os.path.join(gt_path,file),'gt')

        count_face += len(gt_boxes)

        success, predict = image_eval(pred_info, gt_boxes, iou_thresh)

        #print(success)
        #print(predict)

        _img_pr_info = img_pr_info(thresh_num, pred_info, success, predict)

        #print(_img_pr_info)

        pr_curve += _img_pr_info

    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, propose)

    print(ap)
    print(pr_curve)

    fig,ax = plt.subplots()
    ax.plot(recall,propose)
    ax.set(title='Precision-Recall Curve',
           xticks=np.arange(0,1,1/10),
           yticks=np.arange(0,1,1/10),
           xlabel='Recall',
           ylabel='Precision')
    fig.savefig('./test/curve.jpg')