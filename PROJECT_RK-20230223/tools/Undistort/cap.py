import cv2 as cv

video = cv.VideoCapture()
video.open('rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')
cnt = 0

while video.isOpened():
    ret,frame = video.read()
    h,w = frame.shape[:-1]
    if ret is True:
        frame1 = cv.resize(frame,(w//2,h//2))
        cv.imshow('video',frame1)
        c = cv.waitKey(1) & 0xFF

        if c == ord('w'):
            print ('press  WWW ')
            cv.imwrite('images/'+str(cnt)+'.jpg',frame)
            cnt += 1

        elif c == ord('q'):
            break
        
        # del(frame)
    else:
        break