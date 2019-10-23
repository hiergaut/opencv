
import os
from threading import Thread
import paramiko
import sys

home =len(sys.argv) != 1

if home:
    portSsh=50023
    ipRpi='192.168.1.3'
    ipPc='192.168.1.2'
    ncPort=5001
else:

    # portSsh=22
    ipRpi='10.3.141.196'
    # ipRpi='10.3.141.126'
    ipPc='10.3.141.193'
    ncPort=5001

fps=10
h=720
w=1280
# h=1944
# w=2592



def listen():
    # print("listen on")
    os.system("nc -l -p " +str(ncPort) +" | /usr/bin/mplayer -fps " +str(fps) +" -cache 1024 -demuxer h264es -dumpstream -dumpfile /dev/stdout -")
    # print("listen done")

Thread(target=listen).start()

# def openStream():
    # print('stream start')
ssh = paramiko.SSHClient()
ssh.load_system_host_keys()

if home:
    ssh.connect(ipRpi, username='pi', port=str(portSsh))
else:
    ssh.connect(ipRpi, username='pi')
# ssh.exec_command('raspistill -q 100 -h 1080 -w 1920 -o /tmp/img.jpg -t 1 --nopreview')
ssh.exec_command('raspivid -t 0 -fps ' +str(fps) + ' -w ' +str(w) +' -h ' +str(h) +' -o - | nc ' +ipPc +' ' +str(ncPort))
    # ssh.exec_command('raspivid -ih -vs -vf -n -w 1280 -h 720 -o - -t 0 -b 2000000 | nc -v 192.168.1.2 5000')
    # time.sleep(0.5)
    # print("take picture on pi")
    # ssh.exec_command('raspistill -t 1 -o /tmp/img.jpg')
    # print("stream done")


# time.sleep(0.5)
# Thread(target=openStream).start()
# time.sleep(0.5)
