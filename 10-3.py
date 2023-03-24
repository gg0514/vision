import numpy as np
import cv2 as cv
import sys

def construct_yolo_v3():
    f= open('../yolov3/coco.names', 'r')
    class_names=[line.strip() for line in f.readlines()]
    
    model= cv.dnn.readNet('../yolov3/yolov3.weights', '../yolov3/yolov3.cfg')
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    
    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)
    
    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()	# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(100,3))		# 100개 색으로 트랙 구분

from sort import Sort

sort=Sort()

cap=cv.VideoCapture('./slow_traffic_small.mp4')
if not cap.isOpened(): sys.exit('동영상 연결 실패')

while True:
    ret,frame=cap.read()
    if not ret: break
    
    res=yolo_detect(frame,model,out_layers)   
#    persons=[res[i] for i in range(len(res)) if res[i][5]==0] # 부류 0은 사람

    if len(res)==0: 
        tracks=sort.update()
    else:
        tracks=sort.update(np.array(res))
    
    for i in range(len(tracks)):
        x1,y1,x2,y2,track_id=tracks[i].astype(int)
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1+10,y1+40),cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)            

    cv.imshow('Person tracking by SORT',frame)
    
    key=cv.waitKey(1) 
    if key==ord('q'): break 
    
cap.release()		# 카메라와 연결을 끊음
cv.destroyAllWindows()