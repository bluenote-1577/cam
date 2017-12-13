import calibration as cal
import cv2
import numpy as np
import wx
import sys
import time
import math
import pyIGTLink as link

#constant
camNum = 1;
LENGTH_SCALE=10
skipframe=0
minknn=0
minbff=0
posaccuracy = 0.2 
curr_transform = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


def Track(MainFrame,server): 
    sys.stderr.write( "Sending Transforms \n")
    mtx=MainFrame.mtx
    dist=MainFrame.dist
    
    MainFrame.webcam = cv2.VideoCapture(camNum)#to load a file change to:cv2.VideoCapture('file.avi')
    webcam=MainFrame.webcam
    
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    bff = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    t1 = time.time()
    ret, frame = MainFrame.webcam.read() 
    if (ret):
        frame = cal.undistFrame(frame, mtx, dist)
        oldframe = frame
        t0 = t1;
        t1 = time.time()
        ret, frame = webcam.read() 
        frame = cal.undistFrame(frame, mtx, dist)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
        
        
        
        positionsknnRigid = [[.0,0.0]]
        currposknnRigid = [.0,.0]
        positionsbfRigid = [[.0,.0]]
        currposbfRigid = [.0,.0]
        yawbf = [0]
        yawknn = [0]
        
        frame_count = 1
        while(ret):
            skipframe = 0
            #frame counter
            frame_count = frame_count + 1
            #print('frame '+str(frame_count)+':')
                
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
                            #print('map update: waiting for better data')
                            skipframe = 1
                    #not enough matches
                    else: 
                        #print('I\'m Lost.(not enough matches). go back and search!')
                        #TODO: add a method for lost frames so that going back would be easier.
                        skipframe = 1
                        
                    #update the gray image with path    
                    pts2 = np.array(positionsbfRigid, np.int32)+[MainFrame.trackPlayer.Size[1]/2,MainFrame.trackPlayer.Size[0]/2]
                    pts3 = np.array(positionsknnRigid, np.int32)+[MainFrame.trackPlayer.Size[1]/2,MainFrame.trackPlayer.Size[0]/2]
                    trackPad = np.zeros(MainFrame.trackPlayer.DoGetSize(), np.uint8)+125
                    cv2.polylines(trackPad,[pts2],False,(0,0,0))
                    cv2.polylines(trackPad,[pts3],False,(255,255,255))
                    tail=(MainFrame.trackPlayer.Size[1]/10,MainFrame.trackPlayer.Size[0]/10)
                    tip = (tail[0]+int(MainFrame.trackPlayer.Size[1]/10*math.sin(yawbf[len(yawbf)-1])),tail[1]+int(MainFrame.trackPlayer.Size[0]/10.*math.cos(yawbf[len(yawbf)-1])))
                    cv2.arrowedLine(trackPad, tail, tip, (0,0,0), 1)
                    tail=(MainFrame.trackPlayer.Size[1]/10,MainFrame.trackPlayer.Size[0]/10)
                    tip = (tail[0]+int(MainFrame.trackPlayer.Size[1]/10*math.sin(yawknn[len(yawknn)-1])),tail[1]+int(MainFrame.trackPlayer.Size[0]/10.*math.cos(yawknn[len(yawknn)-1])))
                    cv2.arrowedLine(trackPad, tail, tip, (255,255,255), 1)
                    #print(str(currposknnRigid)+', '+str(yawknn[len(yawknn)-1])+', '+str(len(good_matches))+' , ',str(t1-t0))
                    #print(str(currposbfRigid)+', '+str(yawbf[len(yawbf)-1])+', '+str(len(p))+' , ',str(t1-t0))
                else:
                    skipframe=1
            #not enough features
            else:
                print('not enough features to begin with.'+str(len(kp1))+'---'+str(len(oldkp1)))
                print(str(currposbfRigid)+' , ',str(t1-t0))
            
            
            #show the results
            for m in kp1:  
                cv2.circle(gray,(int(m.pt[0]),int(m.pt[1])), 2, (0,0,255), -1)
            
            #show the data  
            rgb=cv2.cvtColor(trackPad,cv2.COLOR_GRAY2RGB)
            rgb=cv2.resize(rgb, MainFrame.trackPlayer.DoGetSize()) 
            image = wx.Bitmap.FromBuffer(MainFrame.trackPlayer.Size[0],MainFrame.trackPlayer.Size[1], rgb)
            MainFrame.trackPlayer.SetBitmap(image)
            
            rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            imageScale=min((MainFrame.btn.GetPosition()[1]-20)*1.0/h,(MainFrame.ClientSize[0]-MainFrame.trackPlayer.Size[0]-10)*1.0/w)
            rgb=cv2.resize(rgb, (int(w*imageScale),int(h*imageScale))) 
            image = wx.Bitmap.FromBuffer(int(w*imageScale),int(h*imageScale), rgb)
            MainFrame.vidPlayer.SetBitmap(image)
            SendTransform(server,currposknnRigid,yawknn[len(yawknn)-1])
            
            if skipframe==1:
                ret, frame = webcam.read()
                frame = cal.undistFrame(frame, mtx, dist) 
                skipframe=0
                
            else:
                oldframe = frame
                t0 = t1;
                t1 = time.time()
                ret, frame = webcam.read() 
                frame = cal.undistFrame(frame, mtx, dist)    

        MainFrame.SetStatusText("Done!")
        
    else:
        MainFrame.SetStatusText("Camera not detected. Tracking did not start.") 
    
    MainFrame.btn.SetLabel('Initialize')
    
    
    
    
def Initialize(MainFrame):
    mtx=MainFrame.mtx
    dist=MainFrame.dist
    
    MainFrame.webcam = cv2.VideoCapture(camNum)#to load a file change to:cv2.VideoCapture('file.avi')
    orb = cv2.ORB_create()
    
    #get the first two frames
    if(MainFrame.webcam.isOpened()):                  
        ret, frame = MainFrame.webcam.read()
        f = cal.undistFrame(frame, mtx, dist)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h,w=gray.shape
    
    i=0    
    while(ret):
        frame = cal.undistFrame(frame, mtx, dist)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1,_  = orb.detectAndCompute(gray,None)
        if len(kp1)!=0 :
            for m in kp1:
                cv2.circle(gray,(int(m.pt[0]),int(m.pt[1])), 2, (0,0,255), -1)
        i=i+1
        if (i==2):
            i=0
            rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            imageScale=min((MainFrame.btn.GetPosition()[1]-20)*1.0/h,(MainFrame.ClientSize[0])*1.0/w)
            rgb=cv2.resize(rgb, (int(w*imageScale),int(h*imageScale))) 
            image = wx.Bitmap.FromBuffer(int(w*imageScale),int(h*imageScale), rgb)
            MainFrame.vidPlayer.SetBitmap(image)
        ret, frame = MainFrame.webcam.read()
   
def SendTransform(server,translation,rotation):
    if server.is_connected():
        rot_a = math.cos(rotation)
        rot_b = math.sin(rotation)

        _data = np.array([[rot_a,-rot_b,0,translation[0] * 15],[rot_b,rot_a,0,translation[1]*15],[0,0,1,0],[0,0,0,1]])
        ultrasound_transform = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        np.dot(_data,ultrasound_transform,out=_data)

        #sys.stderr.write(_data)
        trans_message = link.TransformMessage(_data,device_name="UltraToReference")
        server.add_message_to_send_queue(trans_message)
