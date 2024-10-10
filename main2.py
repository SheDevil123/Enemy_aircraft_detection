from super_gradients.training import models
import cv2
#import supervision
import torch
import time 

#img=cv2.imread("img3.jpg")
save=True
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
print(111)
model=models.get(model_name="yolo_nas_l",num_classes=5,checkpoint_path="E:\\enemy aircraft recognition\\average_classif.pth")
print(111)
cap=cv2.VideoCapture("E:\enemy aircraft recognition\input clips\j10.mp4")

if save:
    fps=cap.get(cv2.CAP_PROP_FPS)
    save_file = cv2.VideoWriter('save.mp4', cv2.VideoWriter_fourcc(*'mp4v'),fps, (int(cap.get(3)),int(cap.get(4))))
print(DEVICE)
start_time=time.time()
while True:
    start_time=time.time()
    _,img=cap.read()
    res=model.predict(img,conf=0.5)
    #print(type(res))
    result=list(res)[0]
    labels=result.prediction.labels.astype(int)
    #print(result.class_names[labels[0]])
    #print(result.class_names)
    boxes=[i.astype(int) for i in result.prediction.bboxes_xyxy]
    confidences=result.prediction.confidence
    print(type(img))
    # print(confidences)
    # print((boxes[0][0],boxes[0][1]),(boxes[0][2],boxes[0][3]))
    for i in range(len(boxes)):
        img=cv2.rectangle(img,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(100,100,100),2)
        conff= format(confidences[i], ".2f") 
        img=cv2.putText(img,f"drone: {conff}",(boxes[i][0],boxes[i][1]-20),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,0),1,cv2.LINE_AA) #result.class_names[labels[i]]

    # time_taken=time.time()-start_time
    # start_time=time.time()
    # t=format(time_taken,".2f")
    # cv2.putText(img,f"prediction_time: {t}",(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)

    if save:
        save_file.write(img)
    cv2.imshow("",img)
    if cv2.waitKey(1)==ord('e'):
        break
cap.release()
