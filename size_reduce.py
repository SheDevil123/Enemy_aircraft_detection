import os
import cv2

img_path=r"C:\Users\Kharghuvel\Downloads\okijuhygtfr\valid\images"
destination=r"C:\Users\Kharghuvel\Downloads\okijuhygtfr\valid\images"


lst=os.listdir(img_path)
print(lst)
os.chdir(destination)

for j,i in enumerate(lst):
    print(i)
    temp=cv2.imread(os.path.join(img_path,i))
    cv2.imshow("",temp)
    cv2.waitKey(20)
    s=(temp.shape[1]//2,temp.shape[0]//2)
    new=cv2.resize(temp,s)
    new=cv2.resize(new,(temp.shape[1],temp.shape[0]))
    print(i)
    cv2.imwrite(f"{i}",new)
    



