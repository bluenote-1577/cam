from operator import sub
import numpy as np
import cv2
import time
import math
import calibration as cal
from ctypes import _endian


ret, mtx, dist = cal.findCamMTX('cal_im/*.jpg')

#constants
LENGTH_SCALE=10
skipframe=0
minknn=0
minbff=0
posaccuracy = 0.2 #To deactivate accuracy make it -1

#initialize the camera
webcam = cv2.VideoCapture(0)#to load a file change to:cv2.VideoCapture('file.avi')

# Initiate the detectors
orb = cv2.ORB_create()

#setting flann parametrers(have not used)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
bff = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#get the first two frames
if(webcam.isOpened()):
    width = int(webcam.get(3)) 
    height = int(webcam.get(4))
    fmps = int(webcam.get(5)) or 20
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('fast.avi',fourcc, fmps, (width,height))
    
    #initialize position
    positionsknnRigid = [[width/2,height/2]]
    currposknnRigid = [width/2,height/2]
    positionsbfRigid = [[width/2,height/2]]
    currposbfRigid = [width/2,height/2]
    yawbf = [0]
    yawknn = [0]
    
    t0 = time.time()
    ret, oldframe = webcam.read()
    oldframe = cal.undistFrame(oldframe, mtx, dist) 
    t1 = time.time()
    ret, frame = webcam.read()
    frame = cal.undistFrame(frame, mtx, dist) 
    


while(ret):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(gray,'press s to start recording',(1,20),3,1,(30,50,60))
    kp1, des1 = orb.detectAndCompute(gray,None)
    if len(kp1)!=0 :
        for m in kp1:
            cv2.circle(gray,(int(m.pt[0]),int(m.pt[1])), 2, (0,0,255), -1)
    
    cv2.imshow('Image',gray)
    oldframe = frame
    t0 = t1;
    t1 = time.time()
    ret, frame = webcam.read() 
    frame = cal.undistFrame(frame, mtx, dist) 
    
    #press q or ESC to get out
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('s') :
        break
        
#continue for all frames as long as they exist
frame_count = 1
while(ret):
    #frame counter
    frame_count = frame_count + 1
    print('frame '+str(frame_count)+':')
        
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    oldgray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
    
    #extract descriptions
    kp1, des1 = orb.detectAndCompute(gray,None)
    oldkp1, olddes1 = orb.detectAndCompute(oldgray,None)
    
    #match the descriptions
    if (olddes1 is not None):
        if(des1 is not None):
            matchesknn = bf.knnMatch(des1,olddes1,k=2)
            matchesbf = bff.knnMatch(des1,olddes1,k=1)
            
            # store all the good matches as per Lowe's ratio test.
            p = []
            oldp = []
            good_matches = []
            for m,n in matchesknn:
                if m.distance < 0.7*n.distance:
                    good_matches.append(m)
                    p.append(kp1[m.queryIdx].pt)
                    oldp.append(oldkp1[m.trainIdx].pt)
            p = np.float32(p)
            oldp = np.float32(oldp)
            
            pbf = []
            oldpbf = []
            for m in matchesbf:
                if len(m)>0:
                    pbf.append(kp1[m[0].queryIdx].pt)
                    oldpbf.append(oldkp1[m[0].trainIdx].pt)
            pbf = np.float32(pbf)
            oldpbf = np.float32(oldpbf)
                
            #for feat in p:
                
            
            #find the frame distance 
            if len(good_matches)>minknn and len(pbf)>minbff:
                # use knn+the rigid transformation between frames
                rigid = cv2.estimateRigidTransform(p, oldp, fullAffine=False)
                if rigid is not None:
                    Yaw_dist_knn = -math.atan(rigid[0,1]/rigid[0,0])
                    changeknnRigid = [(rigid[0,2]*math.cos(Yaw_dist_knn)-rigid[1,2]*math.sin(Yaw_dist_knn))/LENGTH_SCALE,(rigid[0,2]*math.sin(-Yaw_dist_knn)+rigid[1,2]*math.cos(-Yaw_dist_knn))/LENGTH_SCALE]
                    
                # use bf+the rigid transformation between frames
                rigid = cv2.estimateRigidTransform(pbf, oldpbf, fullAffine=False)
                if rigid is not None:
                    Yaw_dist_bf = -math.atan(rigid[0,1]/rigid[0,0])
                    changebfRigid = [(rigid[0,2]*math.cos(Yaw_dist_bf)-rigid[1,2]*math.sin(Yaw_dist_bf))/LENGTH_SCALE,(rigid[0,2]*math.sin(-Yaw_dist_bf)+rigid[1,2]*math.cos(-Yaw_dist_bf))/LENGTH_SCALE]
                
                if  abs((changeknnRigid[0]-changebfRigid[0])/(changeknnRigid[0]+changebfRigid[0]))<posaccuracy and abs((changeknnRigid[1]-changebfRigid[1])/(changeknnRigid[1]+changebfRigid[1]))<posaccuracy:   
                    currposknnRigid = [currposknnRigid[0]+changeknnRigid[0], currposknnRigid[1]+changeknnRigid[1]]
                    currposbfRigid = [currposbfRigid[0]+changebfRigid[0], currposbfRigid[1]+changebfRigid[1]]
                    
                    
                    yawbf.append(yawbf[len(yawbf)-1]+Yaw_dist_bf)
                    yawknn.append(yawknn[len(yawknn)-1]+Yaw_dist_knn)
                    positionsbfRigid.append(currposbfRigid)
                    positionsknnRigid.append(currposknnRigid)
                else:
                    print('map update: waiting for better data')
                    skipframe = 1
            #not enough matches
            else: 
                print('I\'m Lost.(not enough matches). go back and search!')
                #TODO: add a method for lost frames so that going back would be easier.
                skipframe = 1
                
            #update the gray image with path    
            pts2 = np.array(positionsbfRigid, np.int32)
            pts3 = np.array(positionsknnRigid, np.int32)
            cv2.polylines(gray,[pts2],False,(0,0,0))
            cv2.polylines(gray,[pts3],False,(255,255,255))
            print(str(currposknnRigid)+', '+str(yawknn[len(yawknn)-1])+', '+str(len(good_matches))+' , ',str(t1-t0))
            print(str(currposbfRigid)+', '+str(yawbf[len(yawbf)-1])+', '+str(len(p))+' , ',str(t1-t0))
        else:
            skipframe=1
    #not enough features
    else:
        print('not enough features to begin with.'+str(len(kp1))+'---'+str(len(oldkp1)))
        print(str(currposbfRigid)+' , ',str(t1-t0))
    
    
    #show the results   
    cv2.putText(gray,'press q to exit',(1,20),3,1,(30,50,60))
    for m in kp1:  
        cv2.circle(gray,(int(m.pt[0]),int(m.pt[1])), 2, (0,0,255), -1)  
    cv2.imshow('Image',gray)
    
    #press q or ESC to get out
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q') or key == 27:
        break
    
    if skipframe==1:
        ret, frame = webcam.read()
        frame = cal.undistFrame(frame, mtx, dist) 
        skipframe=0
        
    else:
        colorframe = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
        out.write(colorframe)
        oldframe = frame
        t0 = t1;
        t1 = time.time()
        ret, frame = webcam.read() 
        frame = cal.undistFrame(frame, mtx, dist)    
        
    


# When everything done, release the capture
mypts1 = np.array(positionsbfRigid)
myyaw1 = np.array(yawbf)
mypts2 = np.array(positionsknnRigid)
myyaw2 = np.array(yawknn)
np.savetxt("data_bf_knn.csv",np.c_[mypts1,myyaw1,mypts2,myyaw2],delimiter=",")
webcam.release()
out.release()
cv2.destroyAllWindows()
maxx=-1000
maxy=-1000
for val in mypts2:
    if maxx<val[0]:
        maxx=val[0]
    if maxy<val[1]:
        maxy=val[1]
print('bf:')
print(mypts2[len(mypts2)-1][0]-mypts2[0][0])/(maxx-mypts2[0][0])
print(mypts2[len(mypts2)-1][1]-mypts2[0][1])/(maxy-mypts2[0][1])
maxx=-1000
maxy=-1000
for val in mypts1:
    if maxx<val[0]:
        maxx=val[0]
    if maxy<val[1]:
        maxy=val[1]
print('knn:')
print(mypts1[len(mypts1)-1][0]-mypts1[0][0])/(maxx-mypts1[0][0])
print(mypts1[len(mypts1)-1][1]-mypts1[0][1])/(maxy-mypts1[0][1])

