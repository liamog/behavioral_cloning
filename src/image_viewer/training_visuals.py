#!/usr/bin/env python
import wx
import random
import csv
import cv2
import sys

import os
import platform
import numpy as np

IMAGE_COL = 0
SELECTED_COL = 7
ANGLE_COL = 3


class DrawFrame(wx.Frame):

    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        self.image_index = 0
        self.X_samples = []
        self.y_samples = []

        self.CreateMenu()
        self.CreateWidgets()
        self.CreateBindings()
        self.SetupLayout()
        # create the image holder:
        self.Show()

    def CreateMenu(self):
        # Create the menubar
        menuBar = wx.MenuBar()

        # and a menu
        menu = wx.Menu()

        # add an item to the menu, using \tKeyName automatically
        # creates an accelerator, the third param is some help text
        # that will show up in the statusbar
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")

        # bind the menu event to an event handler
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_EXIT)

        # and put the menu on the menubar
        item = wx.MenuItem(menu, wx.ID_ANY, "&Open", "Open file")
        self.Bind(wx.EVT_MENU, self.OnFileOpen, item)
        menu.Append(item)
        self.SetMenuBar(menuBar)
        menuBar.Append(menu, "&File")

    def CreateWidgets(self):
        self.panel = wx.Panel(self)
        self.steering_angle_st = wx.StaticText(self)
        self.iamge_index_st = wx.StaticText(self)
        self.image = wx.StaticBitmap(self)
        self.slider = wx.Slider(self,
                                value=0,
                                minValue=0,
                                maxValue=len(self.X_samples),
                                pos=wx.DefaultPosition,
                                size=wx.DefaultSize,
                                style=wx.SL_HORIZONTAL)

        self.b1 = wx.Button(self, label="<<")
        self.b2 = wx.Button(self, label="<")
        self.b3 = wx.Button(self, label=">")
        self.b4 = wx.Button(self, label=">>")

    def CreateBindings(self):
        self.panel.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)

        self.Bind(wx.EVT_BUTTON, self.OnPrevBig, self.b1)
        self.Bind(wx.EVT_BUTTON, self.OnPrev, self.b2)
        self.Bind(wx.EVT_BUTTON, self.OnNext, self.b3)
        self.Bind(wx.EVT_BUTTON, self.OnNextBig, self.b4)
        self.slider.Bind(wx.EVT_SCROLL, self.OnScrollSlider)

    def SetupLayout(self):
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.Sizer.Add(self.steering_angle_st, 1, wx.CENTER)
        self.Sizer.Add(self.image_index, 1, wx.CENTER)
        self.Sizer.Add(self.image, 1, wx.LEFT)
        self.Sizer.Add(self.slider, 1, wx.EXPAND)

        btnbox = wx.BoxSizer(wx.HORIZONTAL)
        btnbox.Add(self.b1, 0, wx.LEFT | wx.RIGHT, 5)
        btnbox.Add(self.b2, 0, wx.LEFT | wx.RIGHT, 5)
        btnbox.Add(self.b3, 0, wx.LEFT | wx.RIGHT, 5)
        btnbox.Add(self.b4, 0, wx.LEFT | wx.RIGHT, 5)
        self.Sizer.Add(btnbox, 0, wx.CENTER, 5)
        # Set font sizes
        self.steering_angle_st.SetFont(
            wx.FFont(20, wx.FONTFAMILY_SWISS, wx.FONTFLAG_BOLD))
        self.iamge_index_st.SetFont(
            wx.FFont(20, wx.FONTFAMILY_SWISS, wx.FONTFLAG_BOLD))

    def OnClose(self, evt):
        self.Close()

    def OnScrollSlider(self, event):
        self.image_index = event.GetPosition()
        self.Refresh()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()

        if (keycode == 316):
            self.Forward(1)
        if (keycode == 314):
            self.Backward(1)
        event.Skip()

    def OnNext(self, evt):
        self.Forward(1)

    def OnNextBig(self, evt):
        self.Forward(5)

    def Forward(self, skip):
        self.image_index += skip
        if (self.image_index >= len(self.X_samples) - 1):
            self.image_index = len(self.X_samples) - 1
        self.Refresh()

    def OnPrev(self, evt):
        self.Backward(1)

    def OnPrevBig(self, evt):
        self.Backward(5)

    def Backward(self, skip):
        self.image_index -= 1
        if (self.image_index <= 0):
            self.image_index = 0
        self.Refresh()

    def scale_image_to_bitmap(self, image, width, height):
        image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
        result = wx.Bitmap(image)
        return result

    def Refresh(self):
        image_data = self.X_samples[self.image_index]
        image_shape = np.shape(image_data)
        image = wx.EmptyImage(image_shape[1], image_shape[0])
        image.SetData(image_data.tostring())
        bmp = image.ConvertToBitmap()
        self.image.SetBitmap(bmp)

        self.slider.SetValue(self.image_index)
        self.iamge_index_st.SetLabel(
            'Index = {} of {}'.format(
                self.image_index, len(self.X_samples) - 1))

        steering_angle_st_value = self.y_samples[self.image_index]
        self.steering_angle_st.SetLabel(
            'Steering angle= {}'.format(steering_angle_st_value))

    def OnExit(self, evt):
        self.close()

    def OnFileOpen(self, evt):
        path = ""
        dlg = wx.FileDialog(self, "Choose a numpy File",
                            ".", "", "*.npz", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
        else:
            dlg.Destroy()
            return

        dlg.Destroy()
        self.filename = path

        self.LoadData()

    def OnFileSave(self, evt):
        self.SaveData()

    def LoadData(self):
        uncompressed = np.load(self.filename)
        self.X_samples = uncompressed['x']
        self.y_samples = uncompressed['y']

        self.slider.SetMax(len(self.X_samples))
        self.image_index = 0
        self.Refresh()


app = wx.App(False)
F = DrawFrame(None, title="Training data visualization App", size=(700, 700))
app.MainLoop()
