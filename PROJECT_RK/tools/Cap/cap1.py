import cv2

import time
import multiprocessing as mp


def image_put(q, name, pwd, ip, channel=1):

    video_stream_path = 'rtsp://admin:Litian123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'

    cap = cv2.VideoCapture(video_stream_path)
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture(video_stream_path)
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)

def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

def run_multi_camera():
    # user_name, user_pwd = "admin", "password"
    user_name, user_pwd = "admin", "admin123456"
    camera_ip_l = [
        "192.168.35.121",  # ipv4
        "[fe80::3aaf:29ff:fed3:d260]",  # ipv6
        # 把你的摄像头的地址放到这里，如果是ipv6，那么需要加一个中括号。
    ]

    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get, args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()

if __name__ == '__main__':
    run_multi_camera()
