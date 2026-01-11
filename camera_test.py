import cv2
import time

print("Testing camera with YUY2 format...")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed to open camera")
    exit(1)

# Try YUY2 format
fourcc = cv2.VideoWriter_fourcc(*'YUY2')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

print("Camera opened")
time.sleep(2)

for i in range(30):
    ret, frame = cap.read()
    if ret and frame is not None and frame.mean() > 5:
        print(f"YUY2 SUCCESS!")
        cv2.imshow("Working", frame)
        cv2.waitKey(0)
        cap.release()
        exit(0)
    time.sleep(0.1)

print("YUY2 failed, trying NV12...")
cap.release()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
fourcc = cv2.VideoWriter_fourcc(*'NV12')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
time.sleep(2)

for i in range(30):
    ret, frame = cap.read()
    if ret and frame is not None and frame.mean() > 5:
        print(f"NV12 SUCCESS!")
        cv2.imshow("Working", frame)
        cv2.waitKey(0)
        cap.release()
        exit(0)
    time.sleep(0.1)

print("\nAll formats failed.")
print("\nWorkaround: Install OBS Studio and use OBS Virtual Camera")
print("1. Install OBS from https://obsproject.com")
print("2. Add your camera as a source")
print("3. Start Virtual Camera in OBS")
print("4. Select 'OBS Virtual Camera' as camera index 1 or 2")

cap.release()
