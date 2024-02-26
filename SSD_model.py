import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import time
coco_class_names = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}
household_categories = [
    'bottle', 'bowl', 'cup', 'fork', 'knife', 'spoon', 'plate', 'wine glass',
    'clock', 'vase', 'scissors', 'book', 'toothbrush'
]
detector = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/tensorFlow2/variations/fpnlite-320x320/versions/1")
#img = cv2.imread("C:/Users/aadip/PycharmProjects/benchmarking/unnamed.jpg", cv2.IMREAD_COLOR)
#img = Image.open("C:/Users/aadip/PycharmProjects/benchmarking/000000000009.jpg")
failed_detection_list = []
IoU_all_img = []
time_list = []
names_list = []
total_detect = 0
total_Fail = 0
file_path = "C:/Users/aadip/PycharmProjects/benchmarking/images_train/"
num = 0
for file in os.listdir(file_path):
    start = time.time()*1000
    print(file)
    names_list.append(str(file))
    img = cv2.imread("images_train/"+file)
    img_height, img_width, c = img.shape
    #print(img_width)
    #print(img_height)
    img_name = file.split('.')[0]
    #img = img.resize((512,512))
    #img.save("img_new.jpg")
    #cv2.imwrite("./img_new.jpg", img)
    #cv2.waitKey(50)
    #img = img.resize(512, 512)
    image_np = np.array(img, dtype=np.uint8)
    #print(image_np)
    #print(type(image_np))
    #image_np = image_np / 255.0  # Normalize pixel values
    image_np = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    image_np = tf.expand_dims(image_np, axis=0)  # Add batch dimension
    detections = detector(image_np)
    #print(detector_output)
    class_ids = detections["detection_classes"][0].numpy()
    #print(class_ids)
    #print("b")
    bboxes = detections["detection_boxes"][0].numpy()
    #print(bboxes)
    #print("b")
    scores = detections["detection_scores"][0].numpy()
    #bbox_pt = float(bboxes[0][0][0])
    #print(scores)
    bboxes_FRCNN = []
    classes_FRCNN = []
    scores_FRCNN = []
    for i in range(len(detections['detection_scores'][0])):
        score = detections['detection_scores'][0][i].numpy()
        if score > 0.5:  # Adjust threshold as needed
            bbox = detections['detection_boxes'][0][i].numpy()
            #bbox_str = str(bbox)
            #bbox_str = bbox_str[1:-1]
            xmin, ymin, xmax, ymax = bbox
            #ymin, xmin, ymax, xmax = bbox_str.split(" ")
            xmin = float(xmin)#/img_width
            xmax = float(xmax)#/img_width
            ymin = float(ymin)#/img_height
            ymax = float(ymax)#/img_height
            bboxes_FRCNN.append([ymin, xmin, ymax, xmax])
            classes_FRCNN.append(detections['detection_classes'][0][i].numpy())
            scores_FRCNN.append(score)
            #class_id = int(detections['detection_classes'][0][i].numpy())
    #print(detections['num_detections'][0].numpy())
    #print(classes_FRCNN)
    #print('b')
    #print(scores_FRCNN)
    #print('b')
    #print(bboxes_FRCNN)
    names_FRCNN = []
    for clas in classes_FRCNN:
        names_FRCNN.append(coco_class_names[clas])
    i = 0
    #print('g')
    print(names_FRCNN)
    while i < len(names_FRCNN):
        print(i)
        if names_FRCNN[i] not in household_categories:
            names_FRCNN.pop(i)
            bboxes_FRCNN.pop(i)
            scores_FRCNN.pop(i)
        else:
            i += 1
    print("h")
    file_name = os.path.join("real_ann", img_name)+".txt"
    f = open(file_name, "r")
    text_data = f.read()
    f.close()
    names_COCO_str = text_data.split("/n")[0]
    #print(names_COCO_str)
    names_COCO_str = names_COCO_str[1:-1]
    #print(names_COCO_str)
    names_COCO_str_list = names_COCO_str.split(", ")
    names_COCO = []
    for name in names_COCO_str_list:
        names_COCO.append(name[1:-1])
    bboxes_COCO_str = text_data.split("/n")[1]
    #print(bboxes_COCO_str)
    bboxes_COCO_str = bboxes_COCO_str[1:-2]
    #print(bboxes_COCO_str)
    #bboxes_COCO_str = bboxes_COCO_str[0]
    bboxes_COCO_str_list = bboxes_COCO_str.split("], ")
    bboxes_COCO = []
    for bbox_str in bboxes_COCO_str_list:
        bbox_str = bbox_str[1:]
        bbox = bbox_str.split(", ")
        bbox_list = []
        for pt in bbox:
            bbox_list.append(float(pt))
        bboxes_COCO.append(bbox_list)


    #print('b')
    #print(boxes_YOLO)
    #print(names_YOLO)
    #print(names_COCO)

    #normalizing YOLO
    i = len(bboxes_COCO)
    #print(bboxes_COCO)
    #print(type(bboxes_COCO))
    while i > 0:
        i -= 1
        #print(bboxes_COCO[0])
        bboxes_COCO[i][3] = bboxes_COCO[i][3]/img_height
        bboxes_COCO[i][2] = bboxes_COCO[i][2]/img_width
        bboxes_COCO[i][1] = bboxes_COCO[i][1]/img_height
        bboxes_COCO[i][0] = bboxes_COCO[i][0]/img_width
    i = 0
    IOU_lst = []
    fails = 0
    while i < len(names_FRCNN):
        g = 0
        possible_match_list = []
        # print(names_YOLO)
        # print(names_COCO)
        is_in_COCO = 0
        while g < len(names_COCO):
            # print(names_COCO[g])
            # print(type(names_COCO))
            # print(names_YOLO[i])
            # print(type(names_YOLO))
            if names_COCO[g] == names_FRCNN[i]:
                possible_match_list.append(g)
                is_in_COCO = 1
            g += 1
        if is_in_COCO == 0:
            fails += 1
            i += 1
            continue
        length_names = len(names_COCO)
        IoU = 0
        to_be_deleted = 0
        print(possible_match_list)
        print(names_FRCNN[i])
        for index in possible_match_list:
            print(index)
            COCO_x1 = bboxes_COCO[index][0]
            COCO_y1 = bboxes_COCO[index][1]
            COCO_x2 = bboxes_COCO[index][0] + bboxes_COCO[index][2]
            COCO_y2 = bboxes_COCO[index][1] + bboxes_COCO[index][3]
            print([COCO_x1, COCO_y1, COCO_x2, COCO_y2])
            print([bboxes_FRCNN[i][1], bboxes_FRCNN[i][0], bboxes_FRCNN[i][3], bboxes_FRCNN[i][2]])
            if bboxes_FRCNN[i][1] > bboxes_FRCNN[i][3]:
                bboxes_FRCNN[i][1], bboxes_FRCNN[i][3] = bboxes_FRCNN[i][3], bboxes_FRCNN[i][1]
            if bboxes_FRCNN[i][0] > bboxes_FRCNN[i][2]:
                bboxes_FRCNN[i][0], bboxes_FRCNN[i][2] = bboxes_FRCNN[i][2], bboxes_FRCNN[i][0]
            if COCO_x1 > COCO_x2:
                COCO_x1, COCO_x2 = COCO_x2, COCO_x1
            if COCO_y1 > COCO_y2:
                COCO_y1, COCO_y2 = COCO_y2, COCO_y1
            if bboxes_FRCNN[i][1] > COCO_x1:
                intersect_x1 = bboxes_FRCNN[i][1]
            else:
                intersect_x1 = COCO_x1
            if bboxes_FRCNN[i][3] < COCO_x2:
                intersect_x2 = bboxes_FRCNN[i][3]
            else:
                intersect_x2 = COCO_x2
            if bboxes_FRCNN[i][0] > COCO_y1:
                intersect_y1 = bboxes_FRCNN[i][0]
            else:
                intersect_y1 = COCO_y1
            if bboxes_FRCNN[i][2] > COCO_y2:
                intersect_y2 = bboxes_FRCNN[i][2]
            else:
                intersect_y2 = COCO_y2
            intersect_width = intersect_x2 - intersect_x1
            intersect_height = intersect_y2 - intersect_y1
            if intersect_height < 0 or intersect_width < 0:
                continue
            intersection = intersect_width * intersect_height
            COCO_area = (COCO_x2 - COCO_x1) * (COCO_y2 - COCO_y1)
            FRCNN_area = (bboxes_FRCNN[i][3] - bboxes[i][1]) * (bboxes_FRCNN[i][2] - bboxes[i][0])
            union = COCO_area + FRCNN_area
            pos_IoU = intersection/union
            print(intersection)
            print(union)
            print(names_COCO[index])
            print(pos_IoU)
            if pos_IoU > IoU:
                IoU = pos_IoU
                to_be_deleted = index
        bboxes_COCO.pop(to_be_deleted)
        names_COCO.pop(to_be_deleted)
        print('h')
        if IoU > 0.1:
            IOU_lst.append(IoU)
        else:
            fails += 1
        i += 1
    failed_detections = (len(names_COCO) + fails)/length_names
    total_Fail += len(names_COCO) + fails
    total_detect += length_names
    failed_detection_list.append(failed_detections)
    IoU_avg = 0
    for val in IOU_lst:
        IoU_avg += val
    if len(IOU_lst) > 0:
        IoU_avg = IoU_avg/len(IOU_lst)
    else:
        IoU_avg = 0
    IoU_all_img.append(IoU_avg)
    end = time.time()*1000
    time_list.append(int(end - start))
    #if num > 10:
        #break
    num += 1
IoU_all_avg = 0
Iou_len = len(IoU_all_img)
for val in IoU_all_img:
    if val > 0:
        IoU_all_avg += val
    else:
        Iou_len -= 1
if Iou_len > 0:
    IoU_all_avg = IoU_all_avg/Iou_len
failed_detection_avg = total_Fail/total_detect
time_avg = 0
for val in time_list:
    time_avg += val
time_avg = time_avg/len(time_list)
f = open("C:/Users/aadip/PycharmProjects/benchmarking/ssd_Results/results.txt", "w")
f.write(str(IoU_all_img)+"; "+str(failed_detection_list)+"; "+str(time_list)+"\n"+str(IoU_all_avg)+"; "+str(failed_detection_avg)+"; "+str(time_avg)+"\n"+str(names_list))
f.close()
print(IoU_all_img)
print(failed_detection_list)