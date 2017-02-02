#!/usr/bin/env python

"""
Created on 15 January 2017
@author: Vangelis Kostas, Natali Burgos
"""

import sys


from buffer_bci import preproc, FieldTrip
import numpy as np
from keras.models import model_from_json
import struct,socket
import time
from preprocessingEEG import EEGpreprocessing

# Configuration of buffer
buffer_hostname='localhost'
buffer_port=1972

# Configuration of BrainRacer
br_hostname='131.174.105.188'
br_port=5555
br_player=1

# Command offsets, do not change.
CMD_SPEED= 1
CMD_JUMP = 2
CMD_ROLL = 3
CMD_RST  = 99

# Command configuration
CMDS      = [CMD_ROLL, CMD_RST, CMD_JUMP, CMD_SPEED]
THRESHOLDS= [.1,        .1,       .1,     .1]

#Connect to BrainRacers
br_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM);
processor = None

def send_command(command):
    """
    Sends a command to BrainRacer.
    :param command: CMD_SPEED,CMD_JUMP,CMD_ROLL,CMD_RST
    :rtype: None
    """
    global br_socket
    print("Send cmd " + str(command) )
    cmd = (br_player * 10) + command
    data = struct.pack('B', cmd)
    br_socket.sendto(data, (br_hostname, br_port))


def client(hostname='localhost', port=1972, timeout=5000):

    """
    Starting a python client that will be reading the latest signal samples on the buffer
    and using the classifier to get a prediction about the inferred action.
    :param hostname: ip or hostname of the computer running BrainRacers
    :param port: Connection port to the buffer.
    """
    ftc = FieldTrip.Client()
    # Wait until the buffer connects correctly and returns a valid header
    hdr = None;
    while hdr is None :
        print ('Trying to connect to buffer on %s:%i ...'%(hostname,port))
        try:
            ftc.connect(hostname, port)
            print ('\nConnected - trying to read header...')
            hdr = ftc.getHeader()
        except IOError:
                pass
        if hdr is None:
            print('Invalid Header... waiting')
            time.sleep(1)
        else:
            print (hdr)
            print (hdr.labels)

    while True:
        try: latest = ftc.poll()
        except: continue
        data=np.zeros((190,9))
        true_data=ftc.getData(latest)[-190:]
        data[:,:]=true_data[:,:9]
        if processor is None:
            processor=EEGpreprocessing(data.T)
            processor.apply()
        else:
            processor.renew(data.T)
            processor.apply()

        frame_temp=np.zeros((9,19,10))
        TOTAL_X=np.zeros((1,19,90))
        for electrode in range(data.shape[0]):
            frame_temp[electrode]= data[electrode].reshape(int(19),10)
        for frame in range(frame_temp.shape[1]):
            FRAME_Normal=np.zeros((90,))
            for electrode in range(frame_temp.shape[0]):
                FRAME_Normal[electrode*10:(electrode+1)*10] = frame_temp[electrode,frame]
            TOTAL_X[0,frame,:]=FRAME_Normal
        latest_prediction = model.predict(TOTAL_X)[0,-1]
        category = np.argmax(latest_prediction)

        send_command(CMDS[category])

if __name__ == "__main__":
    model_string = open('model/mode.json', 'r').read()
    model = model_from_json(model_string)
    model.load_weights('model/lstm_weigths.h5py')
    hostname=buffer_hostname
    port=buffer_port
    timeout=5000
    if len(sys.argv)>1: # called with options, i.e. commandline
        hostname = sys.argv[1]
    if len(sys.argv)>2:
        try:
            port = int(sys.argv[2])
        except:
            print ('Error: second argument (%s) must be a valid (=integer) port number'%sys.argv[2])
            sys.exit(1)
    client(hostname, port);

