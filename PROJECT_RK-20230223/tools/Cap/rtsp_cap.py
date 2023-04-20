
from time import time
from tkinter.tix import Tree
import cv2 as cv
import cv2
import os

'''
# def run_opencv_camera():
video_stream_path = 'rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'  # local camera (e.g. the front camera of laptop)
cap = cv2.VideoCapture(video_stream_path)

while cap.isOpened():
    is_opened, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()
'''

def test1():
    video = cv.VideoCapture('rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')
    # video.open()
    video.set(cv2.CAP_PROP_BUFFERSIZE,0)
    interval_frames = 10
    cnt = 0
    out_dir = '20220818_out20_baoguang_20'
    os.makedirs(out_dir,exist_ok=True)
    while video.isOpened():
        cnt += 1
        ret,frame = video.read()
        if cnt % interval_frames != 0:
            # del(frame)
            continue
        if frame is None:
            print ('frame NOne')
            continue
        h,w = frame.shape[:-1]
        if ret is True:
            t1 = time()
            # frame1 = cv.resize(frame,(w*2//3,h*2//3))
            # cv.imshow('video',frame1)
            cv.imshow('video',frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # elif key == ord('c'):
            filepath = os.path.join(out_dir,str(cnt)+'.jpg')
            cv2.imwrite(filepath,frame)
            print ('Save ',filepath)

            del(frame)
            t2 = time()
            print ('post time:',t2-t1)
        else:
            break

def test2():
    video = cv.VideoCapture('rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0')
    video.set(cv2.CAP_PROP_BUFFERSIZE,2) 
    video.grab()

    while video.isOpened():
        frame = video.retrieve()
        if frame is not  None:
            t1 = time()
            # frame1 = cv.resize(frame,(w*2//3,h*2//3))
            # cv.imshow('video',frame1)
            cv.imshow('video',frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            del(frame)
            t2 = time()
            print ('post time:',t2-t1)

            
if __name__ == '__main__':
    test1()


