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

  def integrate_reset(self):
    self.sendrecv(':integ:stat stop')
    self.sendrecv(':integ:stat reset')

  def integrate_start(self):
    self.sendrecv(':integ:stat start')

  def integrate_stop(self):
    self.sendrecv(':integ:stat stop')

  def measure_wh_time(self):
    wh, time = self.sendrecv(':meas? wh,time')
    wh = float(wh)
    h, m, s = time.split(',')
    time = int(h) * 3600 + int(m) * 60 + int(s)
    return wh, time

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', action='store_true', help='Initialization.')
  parser.add_argument('-s', action='store_true', help='Start integration.')
  parser.add_argument('-e', action='store_true', help='End integration.')
  parser.add_argument('-r', action='store_true', help='Reset integration.')
  parser.add_argument('-m', action='store_true', help='Measure wh and time.')
  args = parser.parse_args()

  dev = hioki3334()
  if args.i:
    dev.init()
  if args.s:
    dev.integrate_start()
  if args.e:
    dev.integrate_stop()
  if args.r:
    dev.integrate_reset()
  if args.m:
    wh, time = dev.measure_wh_time()
    print(f'{wh:.6f},{time}')

if __name__ == '__main__':
  main()
