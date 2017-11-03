#!/usr/bin/env python
import wx
import random
import csv
import cv2
import sys

import os
import platform

IMAGE_COL = 0
SELECTED_COL = 7
ANGLE_COL = 3


class DrawFrame(wx.Frame):

    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        self.lines = []
        self.selectedlines = []
        self.image_index = 0

        self.CreateMenu()
        self.CreateWidgets()
        self.CreateBindings()
        self.SetupLayout()
        # create the image holder:

        self.Show()
        # self.filename = os.path.abspath(
        #     "../../data/raw/swerving_full copy/driving_log.csv")
        # self.LoadData()

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
        self.Bind(wx.EVT_MENU, self.OnTimeToClose, id=wx.ID_EXIT)

        # and put the menu on the menubar
        item = wx.MenuItem(menu, wx.ID_ANY, "&Open", "Open file")
        self.Bind(wx.EVT_MENU, self.OnFileOpen, item)
        menu.Append(item)
        item = wx.MenuItem(menu, wx.ID_ANY, "&Save", "Save file")
        self.Bind(wx.EVT_MENU, self.OnFileSave, item)
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
                                maxValue=len(self.lines),
                                pos=wx.DefaultPosition,
                                size=wx.DefaultSize,
                                style=wx.SL_HORIZONTAL)

        self.b1 = wx.Button(self, label="<<")
        self.b2 = wx.Button(self, label="<")
        self.b3 = wx.Button(self, label=">")
        self.b4 = wx.Button(self, label=">>")
        self.selected = wx.CheckBox(self, label="Selected")

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
        self.Sizer.Add(self.image, 1, wx.CENTER)
        self.Sizer.Add(self.slider, 1, wx.EXPAND)

        btnbox = wx.BoxSizer(wx.HORIZONTAL)
        btnbox.Add(self.b1, 0, wx.LEFT | wx.RIGHT, 5)
        btnbox.Add(self.b2, 0, wx.LEFT | wx.RIGHT, 5)
        btnbox.Add(self.b3, 0, wx.LEFT | wx.RIGHT, 5)
        btnbox.Add(self.b4, 0, wx.LEFT | wx.RIGHT, 5)
        self.Sizer.Add(btnbox, 0, wx.CENTER, 5)
        self.Sizer.Add(self.selected, 0, wx.CENTER, 5)
        # Set font sizes
        self.steering_angle_st.SetFont(
            wx.FFont(20, wx.FONTFAMILY_SWISS, wx.FONTFLAG_BOLD))
        self.iamge_index_st.SetFont(
            wx.FFont(20, wx.FONTFAMILY_SWISS, wx.FONTFLAG_BOLD))

    def OnTimeToClose(self, evt):
        """Event handler for the button click."""
        print("See ya later!")
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
        if keycode == wx.WXK_SPACE:
            if (self.lines[self.image_index][SELECTED_COL] == 1):
                self.lines[self.image_index][SELECTED_COL] = 0
            else:
                self.lines[self.image_index][SELECTED_COL] = 1
            self.Refresh()
        event.Skip()

    def OnNext(self, evt):
        self.Forward(1)

    def OnNextBig(self, evt):
        self.Forward(5)

    def Forward(self, skip):
        self.image_index += skip
        if (self.image_index >= len(self.lines)):
            self.image_index = 0
        self.Refresh()

    def OnPrev(self, evt):
        self.Backward(1)

    def OnPrevBig(self, evt):
        self.Backward(5)

    def Backward(self, skip):
        self.image_index -= 1
        if (self.image_index == 0):
            self.image_index = len(self.lines) - 1
        self.Refresh()

    def Refresh(self):
        selected = self.lines[self.image_index][SELECTED_COL]
        if selected == 1:
            self.selected.SetValue(wx.CHK_CHECKED)
        else:
            self.selected.SetValue(wx.CHK_UNCHECKED)

        source_path = self.lines[self.image_index][0]
        filename = source_path.split('/')[-1]
        image_path = self.image_dir + filename
        bmp = wx.Bitmap(image_path)
        self.image.SetBitmap(bmp)

        self.iamge_index_st.SetLabel(
            'Index = {}'.format(self.image_index))

        steering_angle_st_value = self.lines[self.image_index][ANGLE_COL]
        self.steering_angle_st.SetLabel(
            'Steering angle= {}'.format(steering_angle_st_value))

    def OnExit(self, evt):
        self.close()

    def OnFileOpen(self, evt):
        path = ""
        dlg = wx.FileDialog(self, "Choose A CSV File",
                            ".", "", "*.csv", wx.FD_OPEN)
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
        self.lines = []
        if not os.path.isfile(self.filename):
            return
        self.image_dir = os.path.dirname(self.filename) + '/IMG/'

        with open(self.filename, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if not line:
                    continue
                # first time opening it add the selected col if necessary
                if(len(line) == 7):
                    line.append(0)
                self.lines.append(line)
        self.slider.SetMax(len(self.lines))
        self.Refresh()

    def SaveData(self):
        with open(self.filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for line in self.lines:
                writer.writerow(line)


app = wx.App(False)
F = DrawFrame(None, title="Image and Angle Selection App", size=(700, 700))
app.MainLoop()
