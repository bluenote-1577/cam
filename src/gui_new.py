#!/usr/bin/env python

# This sample shows how to take advantage of wx.CallAfter when running a
# separate thread and updating the GUI in the main thread
import calibration as cal
import cv2
import numpy as np
import wx
import threading
import time
import math
import gui
import pyIGTLink as link
from sys import platform
import os
import subprocess
import sys

#constant
plusDir = ""
camNum = 1
LENGTH_SCALE=10
skipframe=0
minknn=0
minbff=0
posaccuracy = 0.2
server = link.PyIGTLink(localServer=True,port=22222)

class MainFrame(wx.Frame):

    def __init__(self, parent):
        self.mtx = 0
        self.dist = 0
        
        wx.Frame.__init__(self, parent, title='CallAfter example',size=(900,600),style=wx.SYSTEM_MENU | wx.CAPTION |wx.RESIZE_BORDER)
        self.Center()
                
        self.panel = wx.Panel(self)
        self.btn = wx.Button(self.panel, label="Calibrate")
        self.Quit = wx.Button(self.panel, label="Quit")
        self.gauge = wx.Gauge(self.panel)
        
        self.vidPlayer = wx.StaticBitmap(self.panel)
        self.trackPlayer = wx.StaticBitmap(self.panel,size=(300,300))
        
        self.Bind(wx.EVT_BUTTON, self.OnButton)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        
        self.makeMenuBar()
        self.CreateStatusBar()
        self.SetStatusText("Hi! Please calibrate before start!")
    
    def OnSize(self, event):
        FWidth=self.ClientSize[0]
        FHeight=self.ClientSize[1]
        
        self.panel.SetSize(0, 0, FWidth, FHeight)
        self.btn.SetPosition((self.btn.GetSize()[0],FHeight-self.gauge.GetSize()[1]-self.btn.GetSize()[1]-10))
        self.Quit.SetPosition((FWidth-2*self.Quit.GetSize()[0],FHeight-self.gauge.GetSize()[1]-self.Quit.GetSize()[1]-10))
        self.gauge.SetPosition((0,FHeight-self.gauge.GetSize()[1]))
        self.gauge.SetSize(0, FHeight-self.gauge.GetSize()[1], FWidth, self.gauge.GetSize()[1])
        self.trackPlayer.SetPosition(pt=(self.ClientSize[0]-self.trackPlayer.Size[0],0))
        
        self.panel.Refresh()
        
    def makeMenuBar(self):
        fileMenu = wx.Menu()
        loadItem = fileMenu.Append(-1, "&load...\tCtrl-l","load calibration")
        fileMenu.AppendSeparator()
        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)
        
        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "&Help")

        self.SetMenuBar(menuBar)
        self.Bind(wx.EVT_MENU, self.OnLoad, loadItem)
        self.Bind(wx.EVT_MENU, self.OnExit,  exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)
    
    def OnButton(self, event):
        btn = event.GetEventObject()
        if(btn.GetLabel()=='Quit'):
            self.Close(True)
        
        elif(btn.GetLabel()=='Calibrate'):
            if platform == "linux" or platform == "linux2":
                pathStr = 'cal_im/*.jpg'
            else:
                pathStr = 'cal_im\\*.jpg'

            ret, self.mtx, self.dist = cal.findCamMTX(pathStr)
            if (ret):
                self.SetStatusText("Calibrated! Ready to start!")
                self.btn.SetLabel('Initialize')
            else:
                self.SetStatusText("NOT Calibrated! Please try to calibrate again before start!")
            self.btn.Enable(True) 
            
        elif(btn.GetLabel()=='Initialize'):
            self.gauge.SetValue(70)
            self.SetStatusText("Ready to start!")
            self.btn.SetLabel('Start Tracking') 
            self.btn.Enable(True)
            self.thread = threading.Thread(target=gui.Initialize,args=(self,))
            self.thread.start()

                   
        elif(btn.GetLabel()=='Start Tracking'):
            #plusDir = sys.argv[1]

            #command = plusDir + "/bin/PlusServer --config-file=/home/jim/PlusBuild/test.xml \n "
            #command_list = command.split()
            #subprocess.Popen(command_list)

            #command = plusDir + "/bin/PlusServerRemoteControl --command=START_ACQUISITION --device=CaptureDevice --port=18945\n "
            #command_list = command.split()
            #subprocess.Popen(command_list)

            #command = "sleep 8\n " + plusDir + "/bin/PlusServerRemoteControl --command=STOP_ACQUISITION --device=CaptureDevice --port=18945\n "
            #subprocess.Popen(command,shell=True)

            #command = "../../../PlusBuild/testscript.sh"
            #subprocess.Popen(command,shell=True)
            
            #command = plusDir + "/bin/PlusServerRemoteControl --command=STOP_ACQUISITION --device=CaptureDevice --port=18945\n "
            #command_list = command.split()
            #subprocess.Popen(command_list)

            self.webcam.release()
            self.SetStatusText("Tracking...")
            self.btn.SetLabel('Test') 
            self.btn.Enable(True)
            self.thread = threading.Thread(target=gui.Track,args=(self,server))
            self.thread.start()
                
        elif(btn.GetLabel()=='Test'):
            command = "../../../PlusBuild/testscript.sh"
            subprocess.Popen(command,shell=True) 
            self.btn.SetLabel('Done')
            self.btn.Enable(True)

        elif(btn.GetLabel()=='Done'):
            self.webcam.release()
            self.SetStatusText("Done")
            print()   
           
    def OnExit(self, event):
        self.Close(True)

    def OnLoad(self, event):
        ret, self.mtx, self.dist = cal.findCamMTX('cal_im\\*.jpg')
        if (ret):
            wx.MessageBox("Calibration loaded")
            self.SetStatusText("Calibrated! Ready to start!")
            self.btn.SetLabel('Initialize')
        else:
            wx.MessageBox("Calibration Failed")
            self.SetStatusText("NOT Calibrated! Please try to calibrate again before start!")
        
    def OnAbout(self, event):
        wx.MessageBox("ENPH 459", "About Tracking",wx.OK|wx.ICON_INFORMATION)
    
    
if __name__ == "__main__":
    app = wx.App(0)
    frame = MainFrame(None)
    frame.Show()
    app.MainLoop()
