""" Created on 2023
@author: Sookwang Lee (Korea Aerospace Univ)
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import time
import threading
import math
import socket
import struct
import random
import pickle
import selectors
import subprocess
import argparse
from requests import get
import time
from datetime import datetime, timedelta
import hashlib
import copy
import os

# import python file
from Coordinate import Coordinate
from Neighbor import Neighbour
from NodeBase import NodeBase
from Bootstrap import BootStrap

if __name__ == '__main__':

    this_ip = socket.gethostbyname(socket.gethostname())
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int, default=random.randint(12001, 40000))
    parser.add_argument("--host_port","-P", help="help peer's port number", type=int, default=12000)
    parser.add_argument("--host_addr","-A", help="help peer's ip address", type=str, default='220.67.133.165')
    parser.add_argument("--dimension","-D", help="Node dimension setting", type=int, default=5)
    parser.add_argument("--bootstrap","-B", help="is Bootstrap?" , type=bool, default=False)
    parser.add_argument("--node_num","-N", help="number of node" , type=int, default=1)
    parser.add_argument("--node_nums","-S", help="total number of nodes" , type=int, default=100)
    parser.add_argument("--max_coordinate","-M", help="Coordinate max value", type=int, default=65536)
    parser.add_argument("--hash_text", "-H", help="hash text(subclass name)", type=str, default='animal_cat')
    parser.add_argument("--gpu_num", "-G", help="node GPU number", type=int, default=0)
    args = parser.parse_args()
    
    this_addr = (this_ip, args.port)
    host_addr = (args.host_addr, args.host_port)
    server_list=[]
    server_list = os.environ.get('SERVER_ARRAY').split(',')

    if args.bootstrap:
        NodeBase(args.host_port, args)
    else:
        NodeBase(args.port, args) 
