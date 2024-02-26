# This is a sample Python script.
from ultralytics import YOLO
import os
import cv2
import time
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#using pretrained model to detect objects
model = YOLO('yolov8n.pt')
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
    img_name_YOLO = "image"+str(num)
    results = model.predict(source=img, show=True, conf=0.4, save=True, save_txt=True)
    #getting result values
    prediction_num = "predict"
    if num > 0:
        prediction_num = "predict"+str(num+1)
    file_name_1 = os.path.join("runs", prediction_num, "labels", img_name_YOLO)+".txt"
    f = open("C:/Users/aadip/PycharmProjects/benchmarking/runs/detect/predict/labels/image0.txt", 'r')
    lines = f.readlines()
    f.close()
    f = open("C:/Users/aadip/PycharmProjects/benchmarking/runs/detect/predict/labels/image0.txt", 'w').close()
    boxes_YOLO = []
    classes_YOLO = []
    for line in lines:
        line_list = line.split(" ")
        classes_YOLO.append(line_list[0])
        line_list.pop(0)
        line_list[3] = line_list[3].split("\n")[0]
        boxes_YOLO.append(line_list)
    #print(len(classes_YOLO))
    #print(len(boxes_YOLO))
    #boxes_YOLO = results[0].boxes.xyxy.tolist()
    #classes = results[0].boxes.cls.tolist()
    names = results[0].names
    #print(names)
    names_YOLO = []
    for id in classes_YOLO:
        names_YOLO.append(names[int(id)])
    confidences = results[0].boxes.conf.tolist()
    #print(names_YOLO)
    #finding detections not part of "household objects" list
    household_categories = [
        'bottle', 'bowl', 'cup', 'fork', 'knife', 'spoon', 'plate', 'wine glass',
        'clock', 'vase', 'scissors', 'book', 'toothbrush'
    ]
    values_to_pop = []
    e = len(names_YOLO)
    for i in range(len(names_YOLO)):
        e -= 1
        if names_YOLO[e] not in household_categories:
            values_to_pop.append(e)
    #print(values_to_pop)
    #print(values_to_pop)
    print(names_YOLO)
    print(boxes_YOLO)
    for value in values_to_pop:
        #print(value)
        names_YOLO.pop(value)
        boxes_YOLO.pop(value)
    #getting coco dataset vals
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
    #print(bboxes_COCO)

    #comparing values from YOLO and COCO
    i = 0
    #print("b")
    #print(names_YOLO)
    #print(names_COCO)
    IoU_List = []
    fails = 0
    length_names = len(names_COCO)
    while i < len(boxes_YOLO):
        #print(i)
        #print(len(boxes_YOLO))
        print("f")
        g = 0
        possible_match_list = []
        #print(names_YOLO)
        #print(names_COCO)
        is_in_COCO = 0
        while g < len(names_COCO):
            #print(names_COCO[g])
            #print(type(names_COCO))
            #print(names_YOLO[i])
            #print(type(names_YOLO))
            if names_COCO[g] == names_YOLO[i]:
                possible_match_list.append(g)
                is_in_COCO = 1
            g += 1
        if is_in_COCO == 0:
            fails += 1
            i += 1
            continue
        length_names = len(names_COCO)
        boxes_YOLO[i][0] = float(boxes_YOLO[i][0])
        boxes_YOLO[i][1] = float(boxes_YOLO[i][1])
        boxes_YOLO[i][2] = float(boxes_YOLO[i][2])
        boxes_YOLO[i][3] = float(boxes_YOLO[i][3])
        YOLO_x1 = boxes_YOLO[i][0] - boxes_YOLO[i][2]/2
        YOLO_x2 = boxes_YOLO[i][0] + boxes_YOLO[i][2]/2
        YOLO_y1 = boxes_YOLO[i][1] - boxes_YOLO[i][3]/2
        YOLO_y2 = boxes_YOLO[i][1] + boxes_YOLO[i][3]/2
        IoU = 0
        to_be_deleted = 0
        for index in possible_match_list:
            print("g")
            COCO_x1 = bboxes_COCO[index][0]
            COCO_y1 = bboxes_COCO[index][1]
            COCO_x2 = bboxes_COCO[index][0] + bboxes_COCO[index][2]
            COCO_y2 = bboxes_COCO[index][1] + bboxes_COCO[index][3]
            if COCO_x1 > COCO_x2:
                COCO_x1, COCO_x2 = COCO_x2, COCO_x1
            if COCO_y1 > COCO_y2:
                COCO_y1, COCO_y2 = COCO_y2, COCO_y1
            if YOLO_x1 > YOLO_x2:
                YOLO_x1, YOLO_x2 = YOLO_x2, YOLO_x1
            if COCO_y1 > COCO_y2:
                YOLO_y1, YOLO_y2 = YOLO_y2, YOLO_y1
            if YOLO_x1 > COCO_x1:
                intersect_x1 = YOLO_x1
            else:
                intersect_x1 = COCO_x1
            if YOLO_x2 < COCO_x2:
                intersect_x2 = YOLO_x2
            else:
                intersect_x2 = COCO_x2
            if YOLO_y1 > COCO_y1:
                intersect_y1 = YOLO_y1
            else:
                intersect_y1 = COCO_y1
            if YOLO_y2 > COCO_y2:
                intersect_y2 = YOLO_y2
            else:
                intersect_y2 = COCO_y2
            intersect_width = intersect_x2 - intersect_x1
            intersect_height = intersect_y2 - intersect_y1
            if intersect_height < 0 or intersect_width < 0:
                continue
            intersection = intersect_width * intersect_height
            union = (bboxes_COCO[index][2] * bboxes_COCO[index][3]) + (boxes_YOLO[i][3] * boxes_YOLO[i][2])
            pos_IoU = intersection/union
            if pos_IoU > IoU:
                IoU = pos_IoU
                to_be_deleted = index
        bboxes_COCO.pop(to_be_deleted)
        names_COCO.pop(to_be_deleted)
        #print(bboxes_COCO)
        #print(names_YOLO[i])
        #print(IoU)
        if IoU > 0.1:
            IoU_List.append(IoU)
        else:
            fails += 1
        i += 1
    failed_detections = (len(names_COCO) + fails)/length_names
    total_Fail += len(names_COCO) + fails
    total_detect += length_names
    failed_detection_list.append(failed_detections)
    IoU_avg = 0
    for val in IoU_List:
        IoU_avg += val
    if len(IoU_List) > 0:
        IoU_avg = IoU_avg/len(IoU_List)
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
f = open("C:/Users/aadip/PycharmProjects/benchmarking/Yolo_results/results.txt", "w")
f.write(str(IoU_all_img)+"; "+str(failed_detection_list)+"; "+str(time_list)+"\n"+str(IoU_all_avg)+"; "+str(failed_detection_avg)+"; "+str(time_avg)+"\n"+str(names_list))
f.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/