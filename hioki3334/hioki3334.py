#!/usr/bin/python

import serial
import argparse

class hioki3334:
  def __init__(self):
    self.ser = serial.Serial('/dev/ttyUSB0', timeout=1)

  # return a response message
  def read(self):
    return self.ser.readline().decode().rstrip()

  # send a program message
  def write(self, msg):
    if type(msg) == str:
      msg = msg.encode()
    self.ser.write(msg + b'\n')

  # send a message, receive a response, and parse the confirmation message
  # return data without the confirmation message
  def sendrecv(self, msg):
    self.write(msg)
    res = self.read()
    res = res.split(';')
    if res[-1] != '000':
      pass
      #raise Exception('Error: {}'.format(res[-1]))
    return res[:-1]

  def init(self):
    # send an empty message to end any previous messages
    self.write('')
    # read all response messages to flush out previous responses
    self.ser.readlines()
    # turn on confirmation messages for proper synchronization
    self.sendrecv(':rs232:answ on')
    # turn off header to make parsing easier
    self.sendrecv(':head off')
