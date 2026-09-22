# Face Detection using Haar Cascades with OpenCV and Matplotlib
## Aim
To write a Python program using OpenCV to perform the following image manipulations:

i) Extract ROI from an image.

ii) Perform face detection using Haar Cascades in static images.

iii) Perform eye detection in images.

iv) Perform face detection with label in real-time video from webcam.

## Software Required
Anaconda - Python 3.7 or above

OpenCV library (opencv-python)

Matplotlib library (matplotlib)

Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm
I) Load and Display Images

Step 1: Import necessary packages: numpy, cv2, matplotlib.pyplot

Step 2: Load grayscale images using cv2.imread() with flag 0

Step 3: Display images using plt.imshow() with cmap='gray'

II) Load Haar Cascade Classifiers

Step 1: Load face and eye cascade XML files

III) Perform Face Detection in Images

Step 1: Define a function detect_face() that copies the input image

Step 2: Use face_cascade.detectMultiScale() to detect faces

Step 3: Draw white rectangles around detected faces with thickness 10

Step 4: Return the processed image with rectangles

IV) Perform Eye Detection in Images

Step 1: Define a function detect_eyes() that copies the input image

Step 2: Use eye_cascade.detectMultiScale() to detect eyes

Step 3: Draw white rectangles around detected eyes with thickness 10

Step 4: Return the processed image with rectangles

V) Display Detection Results on Images

Step 1: Call detect_face() or detect_eyes() on loaded images

Step 2: Use plt.imshow() with cmap='gray' to display images with detected regions highlighted

VI) Perform Face Detection on Real-Time Webcam Video

Step 1: Capture video from webcam using cv2.VideoCapture(0)

Step 2: Loop to continuously read frames from webcam

Step 3: Apply detect_face() function on each frame

Step 4: Display the video frame with rectangles around detected faces

Step 5: Exit loop and close windows when ESC key (key code 27) is pressed

Step 6: Release video capture and destroy all OpenCV windows


## PROGRAM
#### Developed by:
#### Name : NAVEEN 
#### Reg No : 212225240098

```
import numpy as np
import cv2 
import matplotlib.pyplot as plt
model = cv2.imread('12(1).jpg',0)
withglass = cv2.imread('12(2).jpg',0)
group = cv2.imread('12(3).jpg',0)
plt.figure(figsize=(20,10))
plt.subplot(131);plt.imshow(cv2.resize(model, (1000, 1000)),cmap='gray');plt.title("Model")
plt.subplot(132);plt.imshow(cv2.resize(withglass, (1000, 1000)),cmap='gray');plt.title("Model with glass")
plt.subplot(133);plt.imshow(cv2.resize(group, (1000, 1000)),cmap='gray');plt.title("Group")
plt.show()
```

```
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(
        f"Face cascade could not be loaded.\nPath: {face_cascade_path}"
    )

print("Face cascade loaded successfully!")
print(face_cascade_path)
```

```
def detect_face(img):
    face_img = img.copy()

    # Convert to grayscale if the image is colored
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    face_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in face_rects:
        cv2.rectangle(
            face_img,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

    return face_img
    result = detect_face(withglass)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face Detection - Model with Glass")
plt.axis('off')
plt.show()
```

```
result = detect_face(model)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Face Detection - Model")
plt.axis('off')
plt.show()
```


```
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if eye_cascade.empty():
    raise RuntimeError(
        f"Eye cascade could not be loaded.\nPath: {eye_cascade_path}"
    )

print("Eye cascade loaded successfully!")
print(eye_cascade_path)
def detect_eyes(img):
    face_img = img.copy()

    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img

    # First detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Detect eyes inside each detected face
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:

            cv2.rectangle(
                face_img,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (255, 255, 255),
                2
            )

    return face_img
    result = detect_eyes(withglass)

plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
plt.title("Eye Detection - Model with Glass")
plt.axis('off')
plt.show()
```

```
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open the camera. "
        "Check whether your webcam is connected or being used by another application."
    )

plt.ion()

fig, ax = plt.subplots(figsize=(10, 7))

ret, frame = cap.read()

if not ret:
    cap.release()
    plt.close(fig)
    raise RuntimeError("Could not read the first frame from the camera.")

frame = detect_face(frame)

im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
ax.set_title("Video Face Detection")
ax.axis('off')

while plt.fignum_exists(fig.number):

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from camera.")
        break

    frame = detect_face(frame)

    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    plt.pause(0.01)

cap.release()
plt.ioff()
plt.close(fig)
```

## OUTPUT

<img width="2167" height="725" alt="rgetgdsfg" src="https://github.com/user-attachments/assets/3290e65e-5f46-431e-a2e6-136336e5c88c" />


## RESULT

Successfully detected and highlighted faces and eyes in images and detected faces in real-time webcam video using OpenCV Haar Cascade classifiers.
