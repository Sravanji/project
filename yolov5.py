import cv2
import torch
import RPi.GPIO as GPIO
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

# Set up YOLOv5
device = select_device('')
model = attempt_load('yolov5s.pt', map_location=device)
stride = int(model.stride.max())

# Set up GPIO for controlling LEDs on the Raspberry Pi
GPIO.setmode(GPIO.BCM)
led_pin = 18  # Change this pin number to the GPIO pin connected to the LED
GPIO.setup(led_pin, GPIO.OUT)

# Load the video
video_path = 'path_to_your_video.mp4'  # Replace this with the path to your video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to tensor
    img = torch.from_numpy(frame).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Detect objects in the frame using YOLOv5
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5)

    # Check for person detection
    if pred[0] is not None:
        det = scale_coords(img.shape[2:], pred[0][:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(pred[0]):
            if int(cls) == 0:  # Person class index
                # If person detected, turn on the LED
                GPIO.output(led_pin, GPIO.HIGH)
                break
        else:
            # If no person detected, turn off the LED
            GPIO.output(led_pin, GPIO.LOW)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
