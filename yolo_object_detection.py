import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def load_image(img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    return img, height, width
def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs
def draw_labels(img, outs, height, width):
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id= np.argmax(scores)
            confidence = scores[class_id]
            if confidence <= 0.5: continue
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            x = int((center_x - w)/2)
            y = int((center_y - h)/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i not in indexes: continue
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img
'''
img, height, width = load_image("sample.jpg")
outs = detect_objects(img)
img = draw_labels(img, outs, height, width)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape
    outs = detect_objects(frame)
    frame = draw_labels(frame, outs, height, width)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()