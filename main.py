import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

lower_color = np.array([100, 150, 50])
upper_color = np.array([140, 255, 255])

canvas = None
prev_center = None

shape_x, shape_y, shape_radius = 100, 100, 30

mode = 'draw'

print("Starting Gesture Control System...")
print("Press 'm' to toggle between Drawing and Moving Shape mode.")
print("Press 'c' to clear the canvas.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 500:  
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
            M = cv2.moments(max_contour)

            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                if mode == 'draw':
                    if prev_center is not None:
                        cv2.line(canvas, prev_center, center, (255, 0, 0), 5)
                    prev_center = center
                elif mode == 'move':
                    shape_x, shape_y = center
                    prev_center = None
        else:
            prev_center = None
    else:
        prev_center = None

    if mode == 'move':
        cv2.circle(frame, (shape_x, shape_y), shape_radius, (0, 255, 255), -1)
        cv2.putText(frame, "Mode: Move Shape", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Mode: Draw Lines", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    frame = cv2.add(frame, canvas)

    cv2.putText(frame, "Press 'm': Switch Mode | 'c': Clear | 'q': Quit", (10, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Gesture Control System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mode = 'move' if mode == 'draw' else 'draw'
        prev_center = None
        print(f"Switched to mode: {mode}")
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        print("Canvas cleared.")

cap.release()
cv2.destroyAllWindows()