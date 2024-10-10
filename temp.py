from super_gradients.training import models
import cv2

# cap=cv2.VideoCapture("E:\enemy aircraft recognition\input clips\j10.mp4")
# _,img=cap.read()
img=cv2.imread("idk3.jpg")
model=models.get(model_name="yolo_nas_l",num_classes=5,checkpoint_path=r"E:\enemy aircraft recognition\average_classif.pth").cuda()


res=model.predict(img,conf=0.3)
result=list(res)[0]
labels=result.prediction.labels.astype(int)
boxes=[i.astype(int) for i in result.prediction.bboxes_xyxy]
confidences=result.prediction.confidence

print(labels,boxes,confidences,result)