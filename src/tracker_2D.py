from operator import sub
import numpy as np
import cv2
import time
import math
from ctypes import _endian

LENGTH_SCALE=10

#initialize the camera
webcam = cv2.VideoCapture(0)#cv2.VideoCapture('3.avi')
num_frames = 3000#int(webcam.get(7))

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
    positionsknnRigid = [[width/2,height/2]]*num_frames
    currposknnRigid = [width/2,height/2]
    positionsknnDis = [[width/2,height/2]]*num_frames
    currposknnDis = [width/2,height/2]
    positionsbfRigid = [[width/2,height/2]]*num_frames
    currpossbfRigid = [width/2,height/2]
    positionsbfDis = [[width/2,height/2]]*num_frames
    currpossbfDis = [width/2,height/2]
    yaw = [0]*num_frames
    
    t0 = time.time()
    ret, oldframe = webcam.read()
    t1 = time.time()
    ret, frame = webcam.read()
    


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
    
    #press q or ESC to get out
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('s') :
        break
        
#continue for all frames as long as they exist
frame_count = 1
while(ret):
    #frame counter
    frame_count = frame_count + 1
    print('frame '+str(frame_count)+' of '+str(num_frames))
        
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    oldgray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
    
    
    #extract descriptions
    kp1, des1 = orb.detectAndCompute(gray,None)
    oldkp1, olddes1 = orb.detectAndCompute(oldgray,None)
    
    #match the descriptions
    if (des1 is not None) and (olddes1 is not None):
        matchesknn = bf.knnMatch(des1,olddes1,k=2)
        matchesbf = bf.match(des1,olddes1)
        
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
            pbf.append(kp1[m.queryIdx].pt)
            oldpbf.append(oldkp1[m.trainIdx].pt)
        pbf = np.float32(pbf)
        oldpbf = np.float32(oldpbf)
            
        
        #find the frame distance 
        if len(good_matches)>10:
            # use knn+the rigid transformation between frames
            rigid = cv2.estimateRigidTransform(p, oldp, fullAffine=False)
            if rigid is not None:
                Yaw_dist = -math.atan(rigid[0,1]/rigid[0,0])
                currposknnRigid = [currposknnRigid[0]+(rigid[0,2]*math.cos(Yaw_dist)-rigid[1,2]*math.sin(Yaw_dist))/LENGTH_SCALE, currposknnRigid[1]+(rigid[0,2]*math.sin(-Yaw_dist)+rigid[1,2]*math.cos(-Yaw_dist))/LENGTH_SCALE]
            
            # use bf+the rigid transformation between frames
            rigid = cv2.estimateRigidTransform(pbf, oldpbf, fullAffine=False)
            if rigid is not None:
                Yaw_dist = -math.atan(rigid[0,1]/rigid[0,0])
                currpossbfRigid = [currpossbfRigid[0]+(rigid[0,2]*math.cos(Yaw_dist)-rigid[1,2]*math.sin(Yaw_dist))/LENGTH_SCALE, currpossbfRigid[1]+(rigid[0,2]*math.sin(-Yaw_dist)+rigid[1,2]*math.cos(-Yaw_dist))/LENGTH_SCALE]
            
            
            #use knn+ the mean change    
            match_dis = []
            for m in good_matches: match_dis.append(map(sub, kp1[m.queryIdx].pt, oldkp1[m.trainIdx].pt))
            frame_dis = [sum(i[0] for i in match_dis)/len(match_dis), sum(i[1] for i in match_dis)/len(match_dis)]
            currposknnDis = [currposknnDis[0]-frame_dis[0]/LENGTH_SCALE, currposknnDis[1]-frame_dis[1]/LENGTH_SCALE]
            
            
            #use bf+ the mean change    
            match_dis = []
            for m in matchesbf: match_dis.append(map(sub, kp1[m.queryIdx].pt, oldkp1[m.trainIdx].pt))
            frame_dis = [sum(i[0] for i in match_dis)/len(match_dis), sum(i[1] for i in match_dis)/len(match_dis)]
            currpossbfDis = [currpossbfDis[0]-frame_dis[0]/LENGTH_SCALE, currpossbfDis[1]-frame_dis[1]/LENGTH_SCALE]
            
            yaw.append(yaw[num_frames-1]+Yaw_dist)
            yaw.pop(0)
            positionsbfDis.append(currpossbfDis)
            positionsbfDis.pop(0)
            positionsbfRigid.append(currpossbfRigid)
            positionsbfRigid.pop(0)
            positionsknnDis.append(currposknnDis)
            positionsknnDis.pop(0)
            positionsknnRigid.append(currposknnRigid)
            positionsknnRigid.pop(0)
        
        #not enough matches
        else: 
            print('not enough matches')
            
        #update the gray image with path    
        pts = np.array(positionsbfDis, np.int32)
        pts1 = np.array(positionsknnDis, np.int32)
        pts2 = np.array(positionsbfRigid, np.int32)
        pts3 = np.array(positionsknnRigid, np.int32)
        #cv2.polylines(gray,[pts],False,(0,255,255))
        #cv2.polylines(gray,[pts1],False,(0,0,0))
        cv2.polylines(gray,[pts2],False,(125,125,125))
        cv2.polylines(gray,[pts3],False,(255,255,255))
        print(yaw[num_frames-1])
        print(str(currpossbfRigid)+', '+str(len(matchesbf))+' , ',str(t1-t0))
    
    #not enough features
    else:
        print('not enough features'+str(len(kp1))+'---'+str(len(oldkp1)))
        print(str(currpossbfRigid)+' , ',str(t1-t0))
    
    #show the results   
    cv2.putText(gray,'press q to exit',(1,20),3,1,(30,50,60))
    for m in kp1:  
        cv2.circle(gray,(int(m.pt[0]),int(m.pt[1])), 2, (0,0,255), -1)  
    cv2.imshow('Image',gray)
    colorframe = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    out.write(colorframe)
    oldframe = frame
    t0 = t1;
    t1 = time.time()
    ret, frame = webcam.read()    
    
    #press q or ESC to get out
    key = cv2.waitKey(1) & 0xFF
    if  key == ord('q') or key == 27:
        break

# When everything done, release the capture

mypts1 = np.array(positionsbfRigid)
mypts2 = np.array(positionsknnRigid)
mypts1.tofile("bf.csv",sep=',',format='%10.5f')
mypts2.tofile("knn.csv",sep=',',format='%10.5f')
webcam.release()
out.release()
cv2.destroyAllWindows()
