import cv2

org_image_path = "/home/bailiangliang/PROJECT_RK-new/data/test_1.png"
dst_image_path = "/home/bailiangliang/PROJECT_RK-new/data/test_1.jpg"
image = cv2.imread(org_image_path, cv2.IMREAD_UNCHANGED)
cv2.imwrite(dst_image_path, image)
