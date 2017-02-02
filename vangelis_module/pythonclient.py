#!/usr/bin/env python
import sys
sys.path.append("../../dataAcq/buffer/python")
sys.path.append( "../../python/signalProc")


from scipy.fftpack import fft
import FieldTrip
import time
import preproc
import numpy as np
from keras.models import model_from_json
import struct,socket
import time

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

# Sends a command to BrainRacer.
def send_command(command):
    global br_socket
    print("Send cmd " + str(command) )
    cmd = (br_player * 10) + command
    data = struct.pack('B', cmd)
    br_socket.sendto(data, (br_hostname, br_port))


def pythonclient(hostname='localhost',port=1972,timeout=5000):

    ftc = FieldTrip.Client()		
    # Wait until the buffer connects correctly and returns a valid header
    hdr = None;
    while hdr is None :
        print 'Trying to connect to buffer on %s:%i ...'%(hostname,port)
        try:
            ftc.connect(hostname, port)
            print '\nConnected - trying to read header...'
            hdr = ftc.getHeader()
	except IOError:
            pass
	
        if hdr is None:
            print 'Invalid Header... waiting'
            time.sleep(1)
        else:
            print hdr
            print hdr.labels

    # Now do the echo server
    nEvents=hdr.nEvents;
    endExpt=None;
    while True:

        data=np.zeros((190,9))
        true_data=ftc.getData()[-190:]

        # print true_data
        # print true_data.shape
        data[:,:]=true_data[:,:9]
        # print true_data

        data = preproc.detrend(data.T)

        frame_temp=np.zeros((9,19,10))
        TOTAL_X=np.zeros((1,19,90))
        for electrode in range(data.shape[0]):
            frame_temp[electrode]= data[electrode].reshape(int(19),10)
        for frame in range(frame_temp.shape[1]):
            FRAME_Normal=np.zeros((90,))
            for electrode in range(frame_temp.shape[0]):
                FRAME=np.absolute(fft(frame_temp[electrode,frame], axis=0))
                FRAME_Normal[electrode*10:(electrode+1)*10] = FRAME
            TOTAL_X[0,frame,:]=FRAME_Normal
        latest_prediction = model.predict(TOTAL_X)[0,-1]
        category = np.argmax(latest_prediction)
        # print latest_prediction
        send_command(CMDS[np.random.randint(0,3,1)[0]])

        # time.sleep(2)

if __name__ == "__main__":


    model_string = open('/media/nat/Data/Documents/BCIpractical/buffer_bci/python/brainhackers/vangelis_module/mode.json','r').read()
    print model_string
    global model
    model = model_from_json(model_string)
    model.load_weights('/media/nat/Data/Documents/BCIpractical/buffer_bci/python/brainhackers/vangelis_module/lstm_weigths.h5py')
    hostname='localhost'
    port=1972
    timeout=5000    
    if len(sys.argv)>1: # called with options, i.e. commandline
        hostname = sys.argv[1]
	if len(sys.argv)>2:
            try:
                port = int(sys.argv[2])
            except:
                print 'Error: second argument (%s) must be a valid (=integer) port number'%sys.argv[2]
                sys.exit(1)
    pythonclient(hostname,port);
