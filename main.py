import cv2
import numpy as np
import time
import tensorflow as tf
import keras

# Path: main.py
# ENVIRONMENT VARIABLES
CAMERA_NUMBER = 2
MODEL = 'Model/yolosegment300ep.h5'
VIDEO = 'videos/a4.avi'
WIDTH = 400
HEIGHT = 400
ZONE_SIZE = 400

label_ids = np.array([0, 1, 2, 3, 4])
label_names = np.array(['10bath', 'airpod', 'allmember', 'eraser', 'fashdrive'])
onehots = np.array(keras.utils.to_categorical(label_ids))
print(onehots)

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU Divice: ", tf.test.gpu_device_name())



cap = cv2.VideoCapture(CAMERA_NUMBER)
# cap = cv2.VideoCapture(VIDEO)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
w = (WIDTH-ZONE_SIZE)//2
h = (HEIGHT-ZONE_SIZE)//2
zone = [(w,h ), (w+ZONE_SIZE, h+ZONE_SIZE)]
green_color = (0, 255, 0)

# Load model
model = keras.models.load_model(MODEL)


def get_label_by_onehot(onehot):
  index_arr = np.where((onehots == onehot).all(axis=1))[0]
  if index_arr.size > 0:
    index = index_arr[0]
    return str(label_names[index])
  return 'None'


def predict_by_image(image):
    img = np.array(cv2.resize(image, [256, 256]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[:, :, 0 ] = gray[:, :]
    img[:, :, 1 ] = gray[:, :]
    img[:, :, 2 ] = gray[:, :]
    img = img / 255.0
    cv2.imshow('Predict',img)
    Z_onehots, Z_pts = model.predict(img[None, :, :, :], verbose = 0)
    # z1 is one hot array have 5 classes
    label = get_label_by_onehot(np.round(Z_onehots[0]))
    if label == 'None':
        return (None, None, None)
    xyxy = np.array(Z_pts*ZONE_SIZE).astype(int)
    # print('predict', Z_onehots[0], Z_pts[0], label)
    return(np.array(np.round(Z_onehots[0])), xyxy[0], label)
    

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if ret == False:
        print("Error: Camera not found")
        break
    f1 = cv2.rectangle(frame.copy(), zone[0], zone[1], green_color, 2)
    image = frame.copy()[h:h+ZONE_SIZE,w:w+ZONE_SIZE]
    (Z_onehots, Z_pts, label) = predict_by_image(image)
    if label != None:
        # print(label, Z_pts)
        # [x1, y1, x2, y2] = z2
        # (x3, y3) = (abs(x2-x1)//2, abs(y2-y1)//2 )
        cv2.rectangle(image, Z_pts, green_color, 1)
        cv2.putText(image, str(label), (Z_pts[0], Z_pts[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, green_color, 1)
    t2 = time.time()
    fps = round(1/(t2-t1),2)
    cv2.putText(f1, 'FPS: '+str(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.imshow('Frame1', f1)
    cv2.imshow('Frame2', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


"""_summary_
    Z_pts_ = Z_pts * 256
    print(Z_pts_[0])    

    for i in range(85, 100):
    img = X_test[i].copy()*255
    img = np.ascontiguousarray(img)

    print(img.shape)
    print(img[100][100])
    xy = np.array(y2_test[i]* 256).astype(int)
    Z_pt = np.array(Z_pts_[i]).astype(int)
    print(xy)
    print(Z_pt)
    lb = str(get_label_by_onehot(np.round(Z_onehots[i])))
    cv2.putText(img, lb, (Z_pt[0], Z_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.rectangle(img, Z_pt, (0, 0, 255), 1)
    cv2.rectangle(img, xy, (0, 255, 0), 1)
    cv2_imshow(img)
"""