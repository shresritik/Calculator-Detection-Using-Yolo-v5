import cv2
import numpy as np
import time
import os
import random
ROOT = os.path.abspath(os.getcwd())
net = cv2.dnn.readNetFromONNX(
    ROOT+"/best.onnx")
file = open(ROOT+"/data.txt", "r")
classes = file.read().split('\n')
fps_start_time = 0
fps = 0
print("Enter Mode \n 1. Image\n 2. Video\n 3. Webcam")

mode = input()
if(mode == "1" or mode == "2"):
    filePath = input("Enter Path of Image/Video: ")
    if(filePath.__contains__("\\")):
        filePath = filePath.replace('\\', '/')
# Base_path = "D:/python/Training of ML/Assignment/calculator detection/output/"
    result_filepath = filePath.split('/')[-1].split('.')[0] + '_yolov5_output'
if(mode == '2'):
    cap = cv2.VideoCapture(
        filePath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        os.makedirs("output/video")
    except:
        pass
    out = cv2.VideoWriter("output/video/"+result_filepath+".mp4",
                          fourcc, int(50), (640, 640))
elif(mode == '3'):
    i = 0
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    try:
        os.makedirs("output/webcam")
    except:
        pass
    out = cv2.VideoWriter("output/webcam/"+f"{random.randint(0,100)}.mp4",
                          fourcc, int(10), (640, 640))
    i += 1
while True:
    if(mode == '1'):
        img = cv2.imread(
            filePath)
    elif(mode == '2' or mode == "3"):
        _, img = cap.read()
    fps_end_time = time.time()
    time_diff = fps_end_time-fps_start_time
    fps = 1/(time_diff)
    fps_start_time = fps_end_time
    fps_text = "FPS: {:.2f}".format(fps)
    if img is None:
        break
    img = cv2.resize(img, (640, 640))
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]
    # print("detections",detections)
    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]
    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        # print("confidence", confidence)
        if confidence > 0.5:
            classes_score = row[5:]

            ind = np.argmax(classes_score)
            if classes_score[ind] > 0:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx - w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                # print("boxes", confidence)
                boxes.append(box)
    indices = cv2.dnn.NMSBoxes(boxes, confidences,  0.25, 0.45)
    # print('indices', indices)
    for i in indices:
        x1, y1, w, h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        print("label", text)

        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
        cv2.putText(img, text, (x1, y1-2),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, fps_text, (5, 30),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

    try:
        os.makedirs("output/images")
    except:
        pass
    if (mode == '2' or mode == '3'):
        out.write(img)
    elif (mode == '1'):
        cv2.imwrite(f"output/images/{result_filepath}.jpg", img)
    cv2.imshow("Frame", img)
    if (mode == '2' or mode == '3'):

        k = cv2.waitKey(1)
    else:
        k = cv2.waitKey(0)

    if k == ord('q'):
        break
if (mode == '2' or mode == '3'):
    cap.release()
    out.release()

# Closes all the frames
cv2.destroyAllWindows()
