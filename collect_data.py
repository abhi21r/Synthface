import cv2

# Replace '0' with the webcam index if you have multiple cameras
cap = cv2.VideoCapture(0)  # Capture video from default webcam

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data = []
count = 0

while count < 100:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_points = classifier.detectMultiScale(gray, 1.3, 5)

    if len(face_points) > 0:
        for (x, y, w, h) in face_points:
            # Extract the face region
            face_frame = frame[y:y + h, x:x + w]

            # Display the face
            cv2.imshow("Face Capture", face_frame)

            # Capture image on key press (space bar)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                data.append(face_frame)
                count += 1
                print(f"Captured image {count} / 100")

    # Display frame with captured image count
    cv2.putText(frame, str(count), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Taking Input", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Release capture and close windows
cap.release()
cv2.destroyAllWindows()

if len(data) == 100:
    name = input("Enter Face holder name: ")
    for i in range(100):
        cv2.imwrite("images/" + name + "_" + str(i) + ".jpg", data[i])
    print("Done")
else:
    print("Not enough images captured. Need more data (100)")
        