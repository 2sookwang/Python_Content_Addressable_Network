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

from Coordinate import Coordinate
from Neighbor import Neighbour

class NodeBase:
    def __init__(self, port, args):
        self.this_ip = socket.gethostbyname(socket.gethostname())
        self.this_addr = (self.this_ip, args.port)
        self.host_addr = (args.host_addr, args.host_port)
        server_list=[]
        server_list = os.environ.get('SERVER_ARRAY').split(',')
        self.dimension = args.dimension
        self.gpu_num = args.gpu_num
        self.node_num = args.node_num
        self.max_coordinate = args.max_coordinate
        self.hash_coord = self.content(self.dimension, self.max_coordinate)
        self.past_queue = []
        #self.hash_coord = self.hash_to_coordinate(args.hash_text, self.dimension, self.max_coordinate, 123)
        if args.bootstrap:
            self.coord_list = []
            for i in range(args.dimension):
                self.coord_list.append(0)
                self.coord_list.append(args.max_coordinate)
            self.c = Coordinate(*self.coord_list)  # Initialized Bootstrap node
            self.n = Neighbour(self.this_addr, self.hash_coord, self.c.coords)  # Neighbout table setting (this_address, hash_coord, coordinate)

            print('Bootstrap hash table :',self.this_addr,self.hash_coord) # Print initialized bootstrap hash coordinate
        else:
            #for i in range(args.dimension):
            #    self.hash_coord[i] = random.randint(0,self.max_coordinate)
            self.port = port
            self.c = None
            self.client_table = dict()
            self.sucess = False
            self.min_addr = None
            self.past_queue = []
            #print("node IP is:", this_ip)
            #print("node Port is:", self.port)
            print('Node hash table :',self.this_addr,self.hash_coord) # Print initialized node hash coordinate

        # socket setting
        self.alive = True

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.this_addr)
        self.socket.setblocking(False)
        self.socket.listen(5)

        # set listening daemon
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        self.listen_t.start()
        
        if args.bootstrap == False :
            self.join()
        self.mainjob()

    def mainjob(self):
        while self.alive:
            time.sleep(5)

    def join(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.host_addr)
        data = ('join', self.this_addr, self.hash_coord)
        sock.sendall(pickle.dumps(data))
        sock.close()

    def run(self):
        """
        thread for listening
        """
        while self.alive:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
            while self.alive:
                for (key, mask) in self.selector.select():
                    key: selectors.SelectorKey
                    srv_sock, callback = key.fileobj, key.data
                    callback(srv_sock, self.selector)
    def accept_handler(self, sock: socket.socket, sel: selectors.BaseSelector):
        """
        accept connection from other nodes
        """
        conn: socket.socket
        conn, addr = self.socket.accept()
        sel.register(conn, selectors.EVENT_READ, self.read_handler)

    def read_handler(self, conn: socket.socket, sel: selectors.BaseSelector):
        """
        read data from other nodes
        """
        message = "---- wait for recv[any other] from {}".format(conn.getpeername())
        recv_data = b""
        while True:
            data_chunk = conn.recv(1024)
            if not data_chunk:  # The loop ends when data reception is complete
                break
            recv_data += data_chunk  # Accumulate received data
        received_data = recv_data
        data =  (received_data)
        time.sleep(0.3)
        self._handle(data, conn)
        data = pickle.loads(received_data)
        threading.Thread(target=self._handle, args=((data,conn)), daemon=True).start()
        sel.unregister(conn)

    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        if data[0] == 'join':
            #print("Node", data[1], "join.")
            # data : ('join', new Node (ip, port), new Node (hash_coord))
            self.node_num += 1
            join_node = data[1]
            join_hash = data[2]
            min_distance = self.max_coordinate ** self.dimension
            for a, [h, c] in self.n.neighbour_table.items():
                d = self.distance(join_hash, h)
                if d < min_distance:
                    min_distance = d
                    self.min_addr = a

            #self.client_table[join_node] = join_hash
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(self.min_addr)
            data = ('coordinate check', join_node, join_hash,self.this_addr, self.past_queue)
            sock.sendall(pickle.dumps(data))
            sock.close()

        elif data[0] == 'coordinate check':
            new_node_addr = data[1]
            new_node_hash = data[2]
            past_addr = data[3]
            self.past_queue = data[4]
            # data : ('coordinate check', new Node (ip, port), new Node (hash_coord), number of node, queue node)
            #print('receive queue node',this_addr, data[4])
            #print(queue_nodes)
            if self.c.isContain(new_node_hash):
                print('This node', self.this_addr,self.c.coords,'is included in the join Hash coordinate!')
                origin_coord, join_coord = self.c.Split_Axis(self.hash_coord, new_node_hash, self.dimension)
                self.c = Coordinate(*list(sum(origin_coord, [])))
                print('-'*10,self.this_addr, 'original Coordinate Changed','by',data[1],'-'*10)
                self.c.show()
                # neighbour_update(peer coord, join node (ip, port), join node coord, join node hash_coord)
                #self.n.neighbour_update(self.c.coords, new_node_addr, join_coord, new_node_hash)
                self.n.neighbour_table[self.this_addr] = [self.hash_coord, self.c.coords]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(new_node_addr)
                data = ('set coordinate', join_coord, self.n.get_neighbour_table(), self.this_addr, self.c.coords, self.hash_coord)
                sock.sendall(pickle.dumps(data))
                sock.close()
                self.n.neighbour_update(self.c.coords)
                self.sucess = True
            else :
                #print('This node is not included in the join Hash coordinate!')
                self.sucess = False
                temp_neighbour = copy.deepcopy(self.n.neighbour_table)
                #del(temp_neighbour[self.this_addr])
                for a , [h,c] in temp_neighbour.items():
                    c = Coordinate(*list(sum(c, [])))
                    if c.isContain(new_node_hash):
                        print('The join Hash coordinate',new_node_hash, 'is included in the spatial coordinate of Address', a, c.coords)
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        #print('send data to', a)
                        sock.connect(a)
                        #data = ('coordinate check', new_node_addr, new_node_hash)
                        sock.sendall(pickle.dumps(data))
                        sock.close()
                        self.sucess = True
                        #self.n.neighbour_table[self.this_addr] = [self.hash_coord, self.c.coords]
                        break
                if self.sucess != True:
                    print('Both this address',self.this_addr,self.c.coords,'and the neighbor node do not include the join hash coordinate.')
                    min_distance = self.max_coordinate ** self.dimension
                    temp_neighbour_table = copy.deepcopy(self.n.neighbour_table)
                    self.past_queue.append(past_addr)
                    for past_node in self.past_queue:
                        if past_node in temp_neighbour_table.keys():
                            del(temp_neighbour_table[past_node])
                    for a, [h,c] in temp_neighbour_table.items():
                        #print(a,h,c)
                        d = self.distance(new_node_hash, h)
                        if d < min_distance:
                            min_distance = d
                            self.min_addr = a
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(self.min_addr)
                    print('Neighbor node with hash coordinate closest to join hash coordinate :', self.min_addr)
                    data = ('coordinate check', new_node_addr, new_node_hash, self.this_addr, self.past_queue)
                    sock.sendall(pickle.dumps(data))
                    sock.close()
                    self.sucess = False

        elif data[0] == 'neighbour update':
            # data : ('neighbour update', New neighbour (ip, port), New neighbour coords, New neighbour hash_coord, Original neighbour address, Original neighbour coords)
            update_neighbour_table = data[1]
            for a, [h, c] in update_neighbour_table.items():
                self.n.neighbour_table[a] = [h, c]
            # self.n.neighbour_table = copy.deepcopy(data[1])
            self.n.neighbour_update(self.c.coords)
            #print(self.this_addr,'update!')
            #print(self.this_addr,'neighbour table:', self.n.neighbour_table.keys(), '\n')

        elif data[0] == 'set coordinate':
            # data : ('set coordinate', join coordinate, neighbour_table (Contain neighbour), address (Contain neighbour), coords (Contain neighbour))
            join_coordinate = data[1]
            contain_neighbour_table = data[2]
            contain_neighbour_addr = data[3]
            contain_neighbour_coord = data[4]
            contain_neighbour_hash = data[5]
            self.c = Coordinate(*list(sum(join_coordinate, [])))
            print('-'*10 ,self.this_addr, 'Set Coordinate', '-'*10)
            self.c.show()
            contain_neighbour_table[self.this_addr] = [self.hash_coord, self.c.coords]
            for a , [h,c] in contain_neighbour_table.items():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(a)
                data = ('neighbour update', contain_neighbour_table)
                sock.sendall(pickle.dumps(data))
                sock.close()
            #contain_neighbour_table[self.this_addr] = [self.hash_coord, self.c.coords]
            contain_neighbour_coords = Coordinate(*list(sum(contain_neighbour_coord, [])))
            self.n = Neighbour(self.this_addr, self.hash_coord, self.c.coords)
            self.n.neighbour_table = copy.deepcopy(contain_neighbour_table)
            self.n.neighbour_update(self.c.coords)
            #time.sleep(3)
            for past_a in self.past_queue:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(past_a)
                data = ('queue reset', self.this_addr)
                sock.sendall(pickle.dumps(data))
                sock.close()
            self.past_queue = []
            log_file = open("./log.txt", "w")
            log_file.write("Queue reset!"+"("+str(self.node_num)+")")
            log_file.close()
            print("done!")

        elif data[0] == 'node scan':
            #print('make file Node{}'.format(args.node_num))
            f = open("/home/deepl/CAN_python/log{}/Node{}.txt".format(datetime.today().strftime("%m%d"), args.node_num),'w')
            f.write(str(self.c.coords)+'\n')
            f.write(str(self.hash_coord)+'\n')
            node_neighbour_table = copy.deepcopy(self.n.neighbour_table)
            for i in range(args.node_nums):
                for server in server_list:
                    if (server, 12000 + i) in self.n.neighbour_table.keys():
                        node_neighbour_table[i] = node_neighbour_table.pop((server, 12000 + i))

            f.write(str(node_neighbour_table)+'\n')
            f.write(str(len(node_neighbour_table)) +'\n')
            f.write(str(self.c.area())+'\n')
            f.close()

        elif data[0] == 'queue reset':
            self.past_queue = []

        elif data[0] == 'start learning':
            #print('All node join complete! Start Fed-Learning')
            Fedlearning(subclass, neighbour_t)

        else:
            a=0

        if conn:
            conn.close

    def hash_to_coordinate(self, identifier, dimensions, max_coordinate, seed):
        self.coordinates = []
        self.seed = seed
        # String creation by combining identifier and seed
        for i in range(dimensions):
            input_string = str(identifier[i]) + str(seed)

            # Convert to bytes using hash function
            hashed_bytes = hashlib.sha256(input_string.encode()).digest()

            # convert bytes to integer
            hashed_int = int.from_bytes(hashed_bytes, byteorder='big')
            coordinate = hashed_int % max_coordinate
            self.coordinates.append(coordinate)
            self.seed += 12

        return self.coordinates

    def distance(self, a, b):
        self.a = a
        self.b = b
        self.d = 0
        for _ in range(len(a)):
            self.d += abs(int(self.a[_]) - int(self.b[_]))
        return self.d

    def content(self, dimensions, max_coordinate):
        ######################### dimension 1 #############################
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            compute = compute_capability[0] + 0.1*compute_capability[1]
        else:
            # no use GPU
            compute = 0
        
        dim1 = int(self.scale_range(compute, 0, 9, self.max_coordinate))

        ######################### dimension 2 #############################
        # Dataset and Transformation Settings
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 64

        # CIFAR-10 dataset load
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

        # ResNet-18 model load
        model = models.resnet18()
        model.fc = torch.nn.Linear(512, 10)  # Set to fit the number of CIFAR-10 classes
        model = model.cuda(self.gpu_num)  # Move the model to the GPU

        # Put the model into inference mode
        model.eval()

        # Measure inference performance
        start_time = time.time()
        num_samples = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(self.gpu_num), labels.cuda(self.gpu_num)
                outputs = model(inputs)
                num_samples += len(inputs)

        end_time = time.time()

        time_taken = end_time - start_time
        throughput = num_samples / time_taken

        # print(f"Throughput: {throughput:.2f} samples/second")
        dim2 = int(self.scale_range(throughput, 0, 10000, self.max_coordinate))

        ######################### dimension 3 #############################
        gpu_properties = torch.cuda.get_device_properties(self.gpu_num)
        gpu_memory = gpu_properties.total_memory / 1024**3
        
        dim3 = int(self.scale_range(gpu_memory, 0, 32, self.max_coordinate))

        ######################### dimension 4 #############################
        dim4 = random.randint(0,self.max_coordinate) 
        
        return [dim1, dim2, dim3, dim4]

    def scale_range(self, input_value, min_value, max_value, max_coordinate):
        # Percentage calculation in current range
        normalized_value = (input_value - min_value) / max_value
    
        # Convert to fit the new scope
        scaled_value = normalized_value * max_coordinate
    
        return scaled_value




