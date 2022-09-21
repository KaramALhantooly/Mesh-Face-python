import cv2
import mediapipe as mp

cap=cv2.VideoCapture(0)

mp_face= mp.solutions.face_mesh
faces= mp_face.FaceMesh()
face_draw = mp.solutions.drawing_utils

drawspec = face_draw.DrawingSpec(thickness=1, circle_radius=1)
while True:
    st , frame = cap.read()
    rgb_frame  =cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

    result = faces.process(rgb_frame)

    if result.multi_face_landmarks is not None:
        
        for face in result.multi_face_landmarks:
            
            face_draw.draw_landmarks(frame , face, mp_face.FACEMESH_CONTOURS ,drawspec)
    cv2.imshow('face detection' , frame)

    if cv2.waitKey(1) & 0xff == ord('x'):
        break
cap.relase()
cv2.destroyAllWindows()
