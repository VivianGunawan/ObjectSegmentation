import os
import cv2
import math
import numpy as np



def load_images_from_folder(folder):
    images = []
    # np.array()
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        #We resize the images for demos just to maximize time 
        #For the original results we use full sized images
        img =cv2.resize(img, (640,400))

        if img is not None:
            images.append(img)
    return images




def detectSkin(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel_intensity = img[i, j]
            B, G, R = pixel_intensity[0], pixel_intensity[1], pixel_intensity[2]
            if ((R > 80 and G > 40 and B > 20) and (max(R, G, B) - min(R, G, B) > 100) and
                    (abs(R - B) > 100) and (R > G) and (R > B)):
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img

def detect_PianoHands():
    images = load_images_from_folder("./input/CS585-PianoImages/")
    # print(len(images))
    addition = images[0]

    for i in range(1, len(images)//2):
        #aggregate the average of 2 images at a time
        addition = cv2.add(addition, images[i])//2
    #we found that the mask worked better when we reduced the brightness abit more
    addition = addition//2
    cv2.imwrite("./output/piano/master_frame.jpg", addition)
    cv2.imwrite("./output/piano/diff1.jpg", cv2.absdiff(addition, images[1]))
    cv2.waitKey(0)
    abs_diff = []
    for i in range(len(images)):
        #get absdiff between each individual image and the aggregate
        diff = cv2.absdiff(images[i], addition)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        #Threshold the image to create a mask
        ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
        #convert back to 3 channel 
        thresh3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("Binary_mask.png", thresh3)
        #apply mask to get ROI
        roi = cv2.bitwise_and(diff, thresh3)
        # cv2.imshow("roi", roi)
        #Use Skin Detection on ROI
        skin = detectSkin(roi)
        if i == 6:
            cv2.imwrite("skin.jpg", skin)
        # cv2.imshow("skin", skin)

        ret, thresh2 = cv2.threshold(skin, 120, 255, 0)
        gray = cv2.cvtColor(thresh2, cv2.COLOR_BGR2GRAY)
        # We find Contours
        contours, hierarchy = cv2.findContours(
            gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        

        cv2.drawContours(images[0], contours, 1, (0, 255, 0),)
        # cv2.imshow("image",image)
        contour_boxes = []
        high_x = 0
        index = 0
        #We filter out contours that are too small and anything past the piano keys
        for j in range(len(contours)):
            area = cv2.contourArea(contours[j])
            if area > 2:
                x, y, w, h = cv2.boundingRect(contours[j])
                if x > high_x:
                    high_x = x
                    index = j
                if x > 350 or x < 255:
                    continue

                contour_boxes.append((x, y, w, h))
        #Drawing the bounding boxes of the contours
        for k in contour_boxes:
            if k == index:
                continue
            else:
                x, y, w, h = k
                images[i] = cv2.rectangle(
                    images[i], (x-10, y-5), (x+w+10, y+h+5), (0, 255, 0), 2)

        # cv2.imshow("skin", skin)
        # cv2.imshow("final", images[i])
        cv2.imwrite(f'output/piano/frame{i}.png', images[i])


def detect_bats(mode):
    frame_buffer_g = load_images_from_folder("./input/CS585-BatImages/Gray/")
    # Use Back Ground Subtractor based on input mode
    if mode == "MOG2":
        bgsub = cv2.createBackgroundSubtractorMOG2()
    elif mode == "KNN":
        bgsub = cv2.createBackgroundSubtractorKNN()
    # Apply the back ground subtractor on the input frames
    bgsub.setHistory(1)
    bgsub.setDetectShadows(False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bats = []
    for i, frame in enumerate(frame_buffer_g):
        fgmask = bgsub.apply(frame)
        frame_ = cv2.adaptiveThreshold(fgmask, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 7, 1)

        # Filter bats in foreground
        dilation = cv2.dilate(fgmask, kernel, iterations=5)
        bats_cnt, _ = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        # draw bounding rectangle and see if the wings are folded or not using compactness
        eps = 0.5
        for cnt in bats_cnt:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
            try:
                bat_area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if (perimeter < eps) or (bat_area < eps):
                    raise ValueError
                circularity = 4*math.pi*(bat_area/(perimeter*perimeter))
                if (0.45 < circularity < 0.55):
                    cv2.putText(frame, "o", (x+3, y-3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "c", (x+3, y-3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            except Exception:
                cv2.putText(frame, "o", (x+3, y-3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                continue
        cv2.drawContours(frame, bats_cnt, -1, (0, 0, 255), 2)
        cv2.imwrite(f"./output/bats/frame{i}.png", frame)
        bats.append(frame)
    return bats


def count_people( mode):
   # Use Back Ground Subtractor based on input mode
    frame_buffer = load_images_from_folder("./input/CS585-PeopleImages/")
    if mode == "MOG2":
        bgsub = cv2.createBackgroundSubtractorMOG2(varThreshold=15)
    elif mode == "KNN":
        bgsub = cv2.createBackgroundSubtractorKNN(dist2Threshold=200)

    # Apply the back ground subtractor on the input frames
    bgsub.setHistory(1)
    bgsub.setDetectShadows(False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    people = []
    for i, frame in enumerate(frame_buffer):
        count = 0
        # Filter people in foreground
        fgmask = bgsub.apply(frame)

        # Filter Foreground noise
        fgmask = cv2.erode(fgmask, kernel, iterations=2)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        # opening = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        fgmask = cv2.medianBlur(fgmask, ksize=3)
        # fgmask = cv.medianBlur(fgmask, ksize=5)
        _, th_people = cv2.threshold(
            fgmask, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Find external contours
        people = []
        contours, _ = cv2.findContours(
            th_people, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < 200:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            people.append(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2, 1)
            count += cnt_area/700
        cv2.putText(frame, str(int(0.2*(int(count))+0.8*(len(people)))),
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.drawContours(frame, people, -1, (0, 0, 255), 2)
        cv2.imwrite(f"./output/people/frame{i}.png", frame)
        people.append(frame)
    return people


if __name__ == "__main__":
    detect_PianoHands()
    # detect_bats("KNN")
    # count_people("KNN")
    

