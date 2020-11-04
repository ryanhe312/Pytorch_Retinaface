import numpy as np
import cv2,time,os,re
import matplotlib.pyplot as plt
from matplotlib import patches

path = './test'

t1 = time.time()

# Initialize the detector cascade.
face_cascade = cv2.CascadeClassifier()

# Load the trained file 
face_cascade.load(cv2.samples.findFile('test/haarcascade_frontalface_alt.xml'))

print('Model Loading Time:',time.time()-t1)

# Path Configure
image_path = os.path.join(path,'Images','000')

if os.path.exists(image_path) is False:
    os.makedirs(image_path)

predict_path = os.path.join(path,'Predicts')

if os.path.exists(predict_path) is False:
    os.makedirs(predict_path)

detect_path = os.path.join(path,'Detects')

if os.path.exists(detect_path) is False:
    os.makedirs(detect_path)

t0 = 0

# testing begin
for file in os.listdir(image_path):
    if re.match('.*(jpg|png|jpeg)',file) is None:
        continue

    t1 = time.time()

    print('Processing file:',file)
    file_path = os.path.join(image_path,file)
    img_raw = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)
    
    img_gray = cv2.equalizeHist(img_gray)
    faces = face_cascade.detectMultiScale(img_gray)

    t2 = time.time()-t1
    t0 += t2
    print('Processing time:',t2)

    dets = np.zeros((len(faces),5))

    for i in range(len(faces)):
        (x,y,w,h) = faces[i]
        dets[i][0] = x
        dets[i][1] = y
        dets[i][2] = x+w
        dets[i][3] = y+h
        dets[i][4] = 1

    # save label
    save_name = os.path.join(predict_path,'.'.join(file.split('.')[:-1])+'.txt')
    with open(save_name, "w") as fd:
        bboxs = dets
        bboxs_num = str(len(bboxs)) + "\n"
        fd.write(bboxs_num)
        for box in bboxs:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            confidence = str(box[4])
            line = str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
            line = line+" " + confidence + " \n"
            fd.write(line)

    # show image
    for b in dets:
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        
    # save image
    output_path = os.path.join(detect_path,file)
    cv2.imwrite(output_path, img_raw)

print('Average Processing time:',t0/len(os.listdir(image_path)))

