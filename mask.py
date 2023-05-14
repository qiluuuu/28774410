import sys
import numpy as np
import cv2

modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    ret = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            ROI = frame[y1:y2, x1:x2].copy()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 框出人脸区域
    ROI = frame[y1:y2, x1:x2].copy()
    hsv_img = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
    lower_hsv_1 = np.array([0, 30, 30])  # 颜色范围低阈值
    upper_hsv_1 = np.array([40, 255, 255])  # 颜色范围高阈值
    lower_hsv_2 = np.array([140, 30, 30])  # 颜色范围低阈值
    upper_hsv_2 = np.array([180, 255, 255])  # 颜色范围高阈值
    mask1 = cv2.inRange(hsv_img, lower_hsv_1, upper_hsv_1)
    mask2 = cv2.inRange(hsv_img, lower_hsv_2, upper_hsv_2)
    mask = mask1 + mask2
    mask = cv2.blur(mask, (3, 3))

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow("mask", mask)
    return ret, frame


if __name__ == '__main__':
    img = cv2.imread("./2.jpg")
    _, result = detectFaceOpenCVDnn(net, img)
    cv2.imshow("face_detection", result)
    cv2.waitKey()
    cv2.destroyAllWindows()





def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def If_Have_Mask(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv_1 = np.array([0, 30, 30])  # 颜色范围低阈值
    upper_hsv_1 = np.array([40, 255, 255])  # 颜色范围高阈值
    lower_hsv_2 = np.array([140, 30, 30])  # 颜色范围低阈值
    upper_hsv_2 = np.array([180, 255, 255])  # 颜色范围高阈值
    mask1 = cv2.inRange(hsv_img, lower_hsv_1, upper_hsv_1)
    mask2 = cv2.inRange(hsv_img, lower_hsv_2, upper_hsv_2)
    mask = mask1 + mask2
    mask = cv2.blur(mask, (3, 3))

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow("mask", mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1:
        return "No Mask"

    contours.sort(key=cnt_area, reverse=True)
    # print(cv2.contourArea(contours[0]))
    area = cv2.contourArea(contours[0])
    mask_rate = area / (img.shape[0] * img.shape[1])
    print(mask_rate)
    if mask_rate < 0.65:
        return "Have Mask"
    else:
        return "No Mask"