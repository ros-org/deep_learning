import cv2
import queue
import time
import threading
q=queue.Queue()
 

rtsp_str = 'rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'
def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture(rtsp_str)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,2)
    # ret, frame = cap.grab() # cap.read()
    success = cap.grab()

    # if (fp % 2 == 0):
    #     continue
    _,frame = cap.retrieve()
    # q.put(frame)
    while success:
        # ret, frame = cap.read()
        # q.put(frame)

        success = cap.grab()
        _,frame = cap.retrieve()

        success = cap.grab()
        _,frame = cap.retrieve()
        q.put(frame)

 
 
def Display():
     print("Start Displaying")
     while True:
         if q.empty() !=True:
            frame=q.get()
            cv2.imshow("frame1", frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
if __name__=='__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()
