#!/usr/bin/env python

from __future__ import division
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from moviepy.editor import VideoFileClip
cwd = os.getcwd()

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

def show_images(images, title, cmap=None):
    cols = 1
    rows = (len(images)+1)//cols
    
    plt.figure(figsize=(15, 10), num=title)
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

test_images = [plt.imread(path) for path in glob.glob('test_images/*.jpg')]

# show_images(test_images, '1. test_images')

### 颜色选择和边缘检测
### Color Selection and Edge Detection

# RGB 颜色空间中的预计图像 
# image is expected be in RGB color space
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([120, 120, 120])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

white_yellow_images = list(map(select_rgb_white_yellow, test_images))
# show_images(white_yellow_images, '2. white_yellow_images')

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray_images = list(map(convert_gray_scale, white_yellow_images))

# show_images(gray_images, '3. gray_images')

def detect_edges(image, low_threshold=50, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)

edge_images = list(map(lambda image: detect_edges(image), gray_images))

# show_images(edge_images, '4. edge_images')

### 确定 ROI
### Identify area of interest

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    # pt_1  = [cols*0.5, rows*0.35]
    # pt_2 = [cols*0.7, rows*0.35]
    # pt_3 = [cols*0.7, rows*0.60]
    # pt_4 = [cols*0.5, rows*0.60]
    pt_1  = [cols*0.39, rows*0.19]
    pt_2 = [cols*0.81, rows*0.19]
    pt_3 = [cols*0.91, rows*0.70]
    pt_4 = [cols*0.28, rows*0.68]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[pt_1, pt_2, pt_3, pt_4]], dtype=np.int32)
    return filter_region(image, vertices)


# 仅显示 ROI 的图像
# images showing the region of interest only
roi_images = list(map(select_region, edge_images))
# roi_images = list(map(select_region, test_images))

show_images(roi_images, '5. roi_images')

### 霍夫线变换
### Hough line transform

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=10, minLineLength=20, maxLineGap=10)
    # return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=10, minLineLength=15, maxLineGap=10)
    # return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=10, minLineLength=20, maxLineGap=5)


# list_of_lines = list(map(hough_lines, roi_images))
list_of_lines = list(map(hough_lines, roi_images))

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(x2 - x1) <= 8 and abs(y2 - y1) >= 10:
                cleaned.append((x1,y1,x2,y2))
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    print("No lines detected: ", len(cleaned))
    return image

# def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
#     # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
#     if make_copy:
#         image = np.copy(image) # don't want to modify the original
#     cleaned = []
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             if abs(y2 - y1) <=1 and abs(x2 - x1) >= 25 and abs(x2 - x1) <= 55:
#                 cleaned.append((x1,y1,x2,y2))
#                 cv2.line(image, (x1, y1), (x2, y2), color, thickness)
#     print("No lines detected: ", len(cleaned))
#     return image


line_images = []
for image, lines in zip(test_images, list_of_lines):
    line_images.append(draw_lines(image, lines))
    
show_images(line_images, '6. line_images')

### 确定矩形停车块
### Identify rectangular blocks of parking

def identify_blocks(image, lines, make_copy=True):
    if make_copy:
        new_image = np.copy(image)

    #Step 1: 创建一个干净的线条列表
    #Step 1: Create a clean list of lines
    cleaned = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if abs(x2 - x1) <= 8 and abs(y2 - y1) >= 10:
                cleaned.append((x1,y1,x2,y2))
    
    #Step 2: 按 x1 位置清理排序
    #Step 2: Sort cleaned by x1 position
    import operator
    list1 = sorted(cleaned, key=operator.itemgetter(0, 1))
    print(list1)
    
    #Step 3: 找到 x1 的聚类在一起 -  clust_dist 分开
    #Step 3: Find clusters of x1 close together - clust_dist apart
    clusters = {}
    dIndex = 0
    clus_dist = 10
    clus_flag = 0

    for i in range(len(list1) - 1):
        distance1 = abs(list1[i+1][0] - list1[i][0])
        distance2 = abs(list1[i+1][1] - list1[i][1])
        distance3 = abs(list1[i+1][2] - list1[i][2])
        distance4 = abs(list1[i+1][3] - list1[i][3])
        
        print('dist: ' + str(distance3))
        if not dIndex in clusters.keys(): clusters[dIndex] = []
        clusters[dIndex].append(list1[i])
        clusters[dIndex].append(list1[i + 1])
        
        # if clus_flag < 2:
        #     if not dIndex in clusters.keys(): clusters[dIndex] = []
        #     clusters[dIndex].append(list1[i])
        #     clusters[dIndex].append(list1[i + 1])

        #     if distance >= clus_dist:
        #         clus_flag += 1
        # else:
        #     dIndex += 1
        #     clus_flag = 0
        ###################################
        # if distance <= clus_dist:
        #     if not dIndex in clusters.keys(): clusters[dIndex] = []
        #     clusters[dIndex].append(list1[i])
        #     clusters[dIndex].append(list1[i + 1])
        # else:
        #     dIndex += 1
    
    #Step 4: 识别此群集周围的矩形坐标
    #Step 4: Identify coordinates of rectangle around this cluster
    rects = {}
    i = 0
    for key in clusters:
        all_list = clusters[key]
        cleaned = list(set(all_list))
        print('len(cleand):' + str(len(cleaned)))
        if len(cleaned) > 3:
            cleaned = sorted(cleaned, key=lambda tup: tup[1])
            prio_x = sorted(cleaned, key=lambda tup: tup[0])
            print(cleaned)
            avg_y1 = cleaned[0][1]
            avg_y2 = cleaned[-1][1]
            avg_x1 = prio_x[0][0]
            avg_x2 = prio_x[-1][0]
            # for tup in prio_x:
            #     avg_x1 += tup[0]
            #     avg_x2 += tup[-1]
            # avg_x1 = avg_x1/len(cleaned)
            # avg_x2 = avg_x2/len(cleaned)
            print('X: '+ str(avg_x1) + ' ' + str(avg_x2))
            print('Y: '+ str(avg_y1) + ' ' + str(avg_y2))
            rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
            i += 1
    
    print("Num Parking Lanes: ", len(rects))

    #Step 5: 在图像上绘制矩形
    #Step 5: Draw the rectangles on the image
    buff = 7
    for key in rects:
        tup_topLeft = (int(rects[key][0] - buff), int(rects[key][1]))
        tup_botRight = (int(rects[key][2] + buff), int(rects[key][3]))
#         print(tup_topLeft, tup_botRight)
        cv2.rectangle(new_image, tup_topLeft,tup_botRight,(0,255,0),3)
    return new_image, rects

# images showing the region of interest only
rect_images = []
rect_coords = []
for image, lines in zip(test_images, list_of_lines):
    new_image, rects = identify_blocks(image, lines)
    rect_images.append(new_image)
    rect_coords.append(rects)
    
show_images(rect_images, '7. rect_images')


### 确定每个地点并计算停车位数量
### Identify each spot and count num of parking spaces

def draw_parking(image, rects, make_copy = True, color=[255, 0, 0], thickness=2, save = True):
    if make_copy:
        new_image = np.copy(image)
    gap = 15.5
    spot_dict = {} # maps each parking ID to its coords
    tot_spots = 0
    adj_y1 = {0: 20, 1:-10, 2:0, 3:-11, 4:28, 5:5, 6:-15, 7:-15, 8:-10, 9:-30, 10:9, 11:-32}
    adj_y2 = {0: 30, 1: 50, 2:15, 3:10, 4:-15, 5:15, 6:15, 7:-20, 8:15, 9:15, 10:0, 11:30}
    
    adj_x1 = {0: -8, 1:-15, 2:-15, 3:-15, 4:-15, 5:-15, 6:-15, 7:-15, 8:-10, 9:-10, 10:-10, 11:0}
    adj_x2 = {0: 0, 1: 15, 2:15, 3:15, 4:15, 5:15, 6:15, 7:15, 8:10, 9:10, 10:10, 11:0}
    cur_len = 0
    for key in rects:
        # Horizontal lines
        tup = rects[key]
        x1 = int(tup[0]+ adj_x1[key])
        x2 = int(tup[2]+ adj_x2[key])
        y1 = int(tup[1] + adj_y1[key])
        y2 = int(tup[3] + adj_y2[key])
        cv2.rectangle(new_image, (x1, y1),(x2,y2),(0,255,0),2)
        num_splits = int(abs(y2-y1)//gap)
        for i in range(0, num_splits+1):
            y = int(y1 + i*gap)
            cv2.line(new_image, (x1, y), (x2, y), color, thickness)
        if key > 0 and key < len(rects) -1 :        
            #draw vertical lines
            x = int((x1 + x2)/2)
            cv2.line(new_image, (x, y1), (x, y2), color, thickness)
        # Add up spots in this lane
        if key == 0 or key == (len(rects) -1):
            tot_spots += num_splits +1
        else:
            tot_spots += 2*(num_splits +1)
            
        # Dictionary of spot positions
        if key == 0 or key == (len(rects) -1):
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                y = int(y1 + i*gap)
                spot_dict[(x1, y, x2, y+gap)] = cur_len +1        
        else:
            for i in range(0, num_splits+1):
                cur_len = len(spot_dict)
                y = int(y1 + i*gap)
                x = int((x1 + x2)/2)
                spot_dict[(x1, y, x, y+gap)] = cur_len +1
                spot_dict[(x, y, x2, y+gap)] = cur_len +2   
    
    print("total parking spaces: ", tot_spots, cur_len)
    if save:
        filename = 'with_parking.jpg'
        cv2.imwrite(filename, new_image)
    return new_image, spot_dict

delineated = []
spot_pos = []
for image, rects in zip(test_images, rect_coords):
    new_image, spot_dict = draw_parking(image, rects)
    delineated.append(new_image)
    spot_pos.append(spot_dict)
    
# show_images(delineated, '8. delineated')

# final_spot_dict = spot_pos[1]
# print(len(final_spot_dict))

# def assign_spots_map(image, spot_dict=final_spot_dict, make_copy = True, color=[255, 0, 0], thickness=2):
#     if make_copy:
#         new_image = np.copy(image)
#     for spot in spot_dict.keys():
#         (x1, y1, x2, y2) = spot
#         cv2.rectangle(new_image, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness)
#     return new_image

# marked_spot_images = list(map(assign_spots_map, test_images))
# # show_images(marked_spot_images, '9. marked_spot_images')

# ### Save spot dictionary as pickle file
# import pickle

# with open('spot_dict.pickle', 'wb') as handle:
#     pickle.dump(final_spot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# def save_images_for_cnn(image, spot_dict = final_spot_dict, folder_name ='for_cnn'):
#     for spot in spot_dict.keys():
#         (x1, y1, x2, y2) = spot
#         (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
#         #crop this image
# #         print(image.shape)
#         spot_img = image[y1:y2, x1:x2]
#         spot_img = cv2.resize(spot_img, (0,0), fx=2.0, fy=2.0) 
#         spot_id = spot_dict[spot]
        
#         filename = 'spot' + str(spot_id) +'.jpg'
#         print(spot_img.shape, filename, (x1,x2,y1,y2))
        
#         cv2.imwrite(os.path.join(folder_name, filename), spot_img)
        
# # save_images_for_cnn(test_images[0])