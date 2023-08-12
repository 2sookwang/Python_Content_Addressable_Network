""" Created on 2023
@author: Sookwang Lee (Korea Aerospace Univ)
"""

import numpy as np
#import torch
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

class Coordinate:
    def __init__(self, *coords):
        # Coordinate : [[dim1_lower, dim1_upper], [dim2_lower, dim2_upper], ....]
        self.coords = [list(coords[i:i+2]) for i in range(0, len(coords), 2)]
        # Coordinate centers : [dim1_center, dim2_center, ...]
        self.centers = [sum(c)/2 for c in self.coords]
        self.lower = [c[0] for c in self.coords]
        self.upper = [c[1] for c in self.coords]
    def show(self):
        print("Node Coordinate:")
        for i in range(len(self.coords)):
            print("Dimension", i+1,":")
            print("Lower:", self.lower[i], "Upper:", self.upper[i])
            print("Center:", self.centers[i])

    def isContain(self, *args):
        # Check if the hash coordinates are included in the peer area.
        hash_coord = list(*args)
        return all(l <= x < u for l, u, x in zip(self.lower, self.upper, hash_coord))

    def isNeighbour(self, c):
        # Check if two peer coordinates are neighbors.
        for i in range(len(self.coords)):
            if self.upper[i] == c.lower[i] or self.lower[i] == c.upper[i]:
                self_half_length = self.upper[i] - self.centers[i] 
                input_half_length = c.upper[i] - c.centers[i]
                center_length = abs(self.centers[i] - c.centers[i])
                if self_half_length + input_half_length == center_length:
                    if all(((self.lower[j] < c.upper[j] <= self.upper[j]) or (self.lower[j] <= c.lower[j] < self.upper[j]) or (self.lower[j] > c.lower[j] and self.upper[j] < c.upper[j])) for j in range(len(self.coords)) if j != i):
                        return True
        return False

    def area(self):
        return np.prod([u-l for l, u in zip(self.lower, self.upper)])

    def isSameSize(self, c):
        # Check if both peers have the same realm size
        return all((u-l) == (c.upper[i]-c.lower[i]) for i, (l, u) in enumerate(zip(self.lower, self.upper)))

    def merge(self, c):
        # Merge each Neighbour to delete coordinate.
        if self.isNeighbour(c) == True:
            merge = [[min(l1, l2), max(u1, u2)] for i, ((l1, u1), (l2, u2)) in enumerate(zip(self.coords, c.coords))]
            return Coordinate(*(lu for axis in merge for lu in axis))

    def Split_Axis(self, origin_hash_map, join_hash_map):
        # Split peer coordinate to join coordinate setting.
        self.oh = origin_hash_map
        self.jh = join_hash_map
        self.dimension = args.dimension
        self.cut = self.centers
        fail = True
        self.hash_max_distance = 0
        self.max_axis = 0

        for axis in range(self.dimension):
            distance = abs(self.oh[axis] - self.jh[axis])
            if distance > self.hash_max_distance:
                self.hash_max_distance = distance
                self.max_axis = axis
             
        while fail:
            # Execute loop to calculate axis coordinates between two hash coordinates.
            if abs(self.oh[self.max_axis] - self.jh[self.max_axis]) == abs(self.oh[self.max_axis] - self.cut[self.max_axis]) + abs(self.jh[self.max_axis] - self.cut[self.max_axis]):
                # if cut axis exist between original hash coordinate with join hash coordinate, split coordinate to coord1, coord2
                coord1 = [list(t) for t in self.coords]
                coord2 = [list(t) for t in self.coords]

                coord1[self.max_axis][1] = self.cut[self.max_axis]
                coord2[self.max_axis][0] = self.cut[self.max_axis]
                fail = False
                # store the dimension of the axis to be swapped
                change_axis = self.max_axis

            if fail:
                # If it is not possible to cut on any axis, designate a random number as the cut coordinate and perform iteratively
                for axis in range(self.dimension):
                    self.cut[self.max_axis] = random.randint(int(self.lower[self.max_axis]), int(self.upper[self.max_axis]+1))
        # Separate the top and bottom of the divided coordinate value and return
        if self.oh[change_axis] - self.jh[change_axis] > 0:
            return coord2, coord1
        else:
            return coord1, coord2

    def getLower(self):
        return self.lower

    def getUpper(self):
        return self.upper

    def hash_distance(self, a, b):
        self.a = a
        self.b = b
        self.d = 0
        for _ in range(len(a)):
            self.d += (int(self.a[_]) - int(self.b[_]))
        self.d = self.d
        return self.d

class NodeBase:
    def __init__(self, port):
        self.dimension = args.dimension
        self.max_coordinate = args.max_coordinate
        self.hash_coord = self.hash_to_coordinate(args.hash_text, self.dimension, self.max_coordinate, 123)
        for i in range(args.dimension):
            self.hash_coord[i] = random.randint(0,self.max_coordinate)
        self.port = port
        self.c = None
        self.client_table = dict()
        self.sucess = False
        self.min_addr = None
        self.past_queue = []
        #print("node IP is:", this_ip)
        #print("node Port is:", self.port)
        print('Node hash table :',this_addr,self.hash_coord) # Print initialized node hash coordinate
        
        # socket setting
        self.alive = True

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(this_addr)
        self.socket.setblocking(False)
        self.socket.listen(5)

        # set listening daemon
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        self.listen_t.start()

        self.join()
        self.mainjob()

    def mainjob(self):
        while self.alive:
            time.sleep(5)

    def join(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(host_addr)
        data = ('join', this_addr, self.hash_coord)
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
            #self.node_num += 1
            join_node = data[1]
            join_hash = data[2]
            min_distance = args.max_coordinate ** args.dimension
            for a, [h, c] in self.n.neighbour_table.items():
                d = self.distance(join_hash, h)
                if d < min_distance:
                    min_distance = d
                    self.min_addr = a

            self.client_table[join_node] = join_hash
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(self.min_addr)
            data = ('coordinate check', join_node, join_hash,this_addr, self.past_queue)
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
                print('This node', this_addr,self.c.coords,'is included in the join Hash coordinate!')
                origin_coord, join_coord = self.c.Split_Axis(self.hash_coord, new_node_hash)
                self.c = Coordinate(*list(sum(origin_coord, [])))
                print('-'*10,this_addr, 'original Coordinate Changed','by',data[1],'-'*10)
                self.c.show()
                # neighbour_update(peer coord, join node (ip, port), join node coord, join node hash_coord)
                #self.n.neighbour_update(self.c.coords, new_node_addr, join_coord, new_node_hash)
                self.n.neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(new_node_addr)
                data = ('set coordinate', join_coord, self.n.get_neighbour_table(), this_addr, self.c.coords, self.hash_coord, self.node_num)
                sock.sendall(pickle.dumps(data))
                sock.close()
                self.n.neighbour_update(self.c.coords)
                self.sucess = True

            else :
                #print('This node is not included in the join Hash coordinate!')
                self.sucess = False
                temp_neighbour = copy.deepcopy(self.n.neighbour_table)
                #del(temp_neighbour[this_addr])
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
                        #self.n.neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
                        break
                if self.sucess != True:
                    print('Both this address',this_addr,self.c.coords,'and the neighbor node do not include the join hash coordinate.')
                    min_distance = args.max_coordinate ** args.dimension
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
                    data = ('coordinate check', new_node_addr, new_node_hash, this_addr, self.past_queue)
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
            #print(this_addr,'update!')
            #print(this_addr,'neighbour table:', self.n.neighbour_table.keys(), '\n')
 
        elif data[0] == '_neighbour update':
            # data = ('_neighbour update', new neighbour address, new neighbour coordinate, new neighbour hash_coord)
            _new_neighbour_addr = data[1]
            _new_neighbour_coord = data[2]
            _new_neighbour_hash = data[3]
            self.n.neighbour_table[_new_neighbour_addr] = [_new_neighbour_hash, _new_neighbour_coord]

        elif data[0] == 'set coordinate':
            # data : ('set coordinate', join coordinate, neighbour_table (Contain neighbour), address (Contain neighbour), coords (Contain neighbour), number of node)
            join_coordinate = data[1]
            contain_neighbour_table = data[2]
            contain_neighbour_addr = data[3]
            contain_neighbour_coord = data[4]
            contain_neighbour_hash = data[5]
            self.node_num = data[6]
            self.c = Coordinate(*list(sum(join_coordinate, [])))
            print('-'*10 ,this_addr, 'Set Coordinate', '-'*10)
            self.c.show()
            contain_neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
            for a , [h,c] in contain_neighbour_table.items():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(a)
                data = ('neighbour update', contain_neighbour_table)
                sock.sendall(pickle.dumps(data))
                sock.close()
            #contain_neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
            contain_neighbour_coords = Coordinate(*list(sum(contain_neighbour_coord, [])))
            self.n = Neighbour(this_addr, self.hash_coord, self.c.coords)
            self.n.neighbour_table = copy.deepcopy(contain_neighbour_table)
            self.n.neighbour_update(self.c.coords)
            #time.sleep(3)
            for past_a in self.past_queue:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(past_a)
                data = ('queue reset', this_addr)
                sock.sendall(pickle.dumps(data))
                sock.close()
            self.past_queue = []
            log_file = open("/home/deepl/CAN_python/log.txt", "w")
            log_file.write("Queue reset!"+"("+str(args.node_num)+")")
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

class BootStrap(NodeBase):
    def __init__(self, port):
        self.dimension = args.dimension
        self.max_coordinate = args.max_coordinate
        self.hash_coord = self.hash_to_coordinate(args.hash_text, self.dimension, self.max_coordinate, 123)
        for i in range(args.dimension):
            self.hash_coord[i] = random.randint(0,self.max_coordinate)
        self.sucess = False
        # Initialized client table
        self.client_table = dict()
        self.past_queue = []
        # Add Bootstrap node address to client table
        self.client_table[this_addr] = self.hash_coord 
        self.node_num = 0
        self.node_nums = args.node_nums
        self.min_addr = None

        #print("------------Bootstrap Node initialized----------------")
        self.coord_list = []
        for i in range(args.dimension):
            self.coord_list.append(0)
            self.coord_list.append(args.max_coordinate)
        self.c = Coordinate(*self.coord_list)  # Initialized Bootstrap node
        #self.c.show()
        self.n = Neighbour(this_addr, self.hash_coord, self.c.coords)  # Neighbout table setting (this_address, hash_coord, coordinate)

        print('Bootstrap hash table :',this_addr,self.hash_coord) # Print initialized bootstrap hash coordinate

        # Socket Communication setting
        self.alive = True

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(this_addr)
        self.socket.setblocking(False)
        self.socket.listen(5)

        # set listening daemon
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")

        self.listen_t.start()

        self.mainjob()

    def mainjob(self):
        while self.alive:
            time.sleep(5)
            #print("bootstrap coord:",self.get_coordinate())

    def get_coordinate(self):
        return self.c.coords

    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        if data[0] == 'join':
            #print("Node", data[1], "join.")
            self.node_num += 1
            # data : ('join', new Node (ip, port), new Node (hash_coord))
            join_node = data[1]
            join_hash = data[2]
            if self.c.isContain(join_hash):
                print('This node', this_addr,self.c.coords,'is included in the join Hash coordinate!')
                origin_coord, join_coord = self.c.Split_Axis(self.hash_coord, join_hash)
                self.c = Coordinate(*list(sum(origin_coord, [])))
                print('-'*10,this_addr ,'original Coordinate Changed','by',data[1],'-'*10)
                self.c.show()
                # neighbour_update(peer coord, join node (ip, port), join node coord, join node hash_coord)
                #self.n.neighbour_update(self.c.coords, new_node_addr, join_coord, new_node_hash)
                self.n.neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(join_node)
                data = ('set coordinate', join_coord, self.n.get_neighbour_table(), this_addr, self.c.coords, self.hash_coord, self.node_num)
                sock.sendall(pickle.dumps(data))
                sock.close()
                self.n.neighbour_update(self.c.coords)
            else:
                min_distance = args.max_coordinate ** args.dimension
                for a, [h, c] in self.n.neighbour_table.items():
                    d = self.distance(join_hash, h)
                    if d < min_distance:
                        min_distance = d
                        self.min_addr = a

                self.client_table[join_node] = join_hash
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(self.min_addr)
                data = ('coordinate check', join_node, join_hash, this_addr, self.past_queue)
                sock.sendall(pickle.dumps(data))
                sock.close()

        elif data[0] == 'coordinate check':
            new_node_addr = data[1]
            new_node_hash = data[2]
            past_addr = data[3]
            self.past_queue = data[4]
            temp_client = copy.deepcopy(self.client_table)
            # data : ('coordinate check', new Node (ip, port), new Node (hash_coord), number of node, queue node)
            #print('receive queue node',this_addr, data[4])
            queue_nodes = data[4]
            if type(queue_nodes) is tuple:
                self.queue.append(queue_nodes)
            else:
                for i in range(len(queue_nodes)):
                    self.queue.append(queue_nodes[i])

            #print(queue_nodes)
            if self.c.isContain(new_node_hash):
                print('This node', this_addr,self.c.coords,'is included in the join Hash coordinate!')
                origin_coord, join_coord = self.c.Split_Axis(self.hash_coord, new_node_hash)
                self.c = Coordinate(*list(sum(origin_coord, [])))
                print('-'*10,this_addr ,'original Coordinate Changed','by',data[1],'-'*10)
                self.c.show()
                # neighbour_update(peer coord, join node (ip, port), join node coord, join node hash_coord)
                #self.n.neighbour_update(self.c.coords, new_node_addr, join_coord, new_node_hash)
                self.n.neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(new_node_addr)
                data = ('set coordinate', join_coord, self.n.get_neighbour_table(), this_addr, self.c.coords, self.hash_coord, self.node_num)
                sock.sendall(pickle.dumps(data))
                sock.close()
                self.n.neighbour_update(self.c.coords)
                self.sucess = True

            else :
                #print('This node is not included in the join Hash coordinate!')
                self.sucess = False
                temp_neighbour = copy.deepcopy(self.n.neighbour_table)
                #del(temp_neighbour[this_addr])
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
                        #self.n.neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
                        break
                if self.sucess != True:
                    print('Both this address',this_addr,self.c.coords,'and the neighbor node do not include the join hash coordinate.')
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
                    data = ('coordinate check', new_node_addr, new_node_hash, this_addr, self.past_queue)
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
            #print(this_addr,'update!')
            #print(this_addr,'neighbour table:',self.n.neighbour_table.keys(),'\n')

        elif data[0] == '_neighbour update':
            # data = ('_neighbour update', new neighbour address, new neighbour coordinate, new neighbour hash_coord)
            _new_neighbour_addr = data[1]
            _new_neighbour_coord = data[2]
            _new_neighbour_hash = data[3]
            self.n.neighbour_table[_new_neighbour_addr] = [_new_neighbour_hash, _new_neighbour_coord]

        elif data[0] == 'set coordinate':
            # data : ('set coordinate', join coordinate, neighbour_table (Contain neighbour), address (Contain neighbour), coords (Contain neighbour), number of node)
            join_coordinate = data[1]
            contain_neighbour_table = data[2]
            contain_neighbour_addr = data[3]
            contain_neighbour_coord = data[4]
            contain_neighbour_hash = data[5]
            self.node_num = data[6]
            self.c = Coordinate(*list(sum(join_coordinate, [])))
            print('-'*10 ,this_addr ,'Set Coordinate', '-'*10)
            self.c.show()
            contain_neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
            for a , [h,c] in contain_neighbour_table.items():
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(a)
                data = ('neighbour update', contain_neighbour_table)
                sock.sendall(pickle.dumps(data))
                sock.close()
            #contain_neighbour_table[this_addr] = [self.hash_coord, self.c.coords]
            contain_neighbour_coords = Coordinate(*list(sum(contain_neighbour_coord, [])))
            self.n = Neighbour(this_addr, self.hash_coord, self.c.coords)
            self.n.neighbour_table = copy.deepcopy(contain_neighbour_table)
            self.n.neighbour_update(self.c.coords)
            #time.sleep(3)
            for past_a in self.past_queue:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(past_a)
                data = ('queue reset', this_addr)
                sock.sendall(pickle.dumps(data))
                sock.close()
            self.past_queue = []
            print('Queue reset!')

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
            self.d += (self.a[_] - self.b[_])
        self.d = self.d
        return self.d

class Neighbour:
    def __init__(self, address, hash_map, coordinate):
        self.addr = address
        self.neighbour_table = dict()
        self.hash_coord = hash_map
        self.coords = coordinate

        self.neighbour_table[self.addr] = [self.hash_coord, self.coords]

    def get_hash_coord(self):
        return self.hash_coord

    def get_address(self):
        return self.addr

    def get_coordinate(self, address):
        return self.neighbour_table[address]

    def get_neighbour_table(self):
        return self.neighbour_table

    def neighbour_update(self, coordinates):
        node_coordinate = Coordinate(*list(sum(coordinates,[])))
        remove_addr = []
        for a, [h,c] in self.neighbour_table.items():
            n_c = Coordinate(*list(sum(c,[])))
            if node_coordinate.isNeighbour(n_c) == False:
                remove_addr.append(a)
        for k in remove_addr:
            del(self.neighbour_table[k])
        return self.neighbour_table

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
    args = parser.parse_args()
    
    this_addr = (this_ip, args.port)
    host_addr = (args.host_addr, args.host_port)
    server_list=[]
    server_list = os.environ.get('SERVER_ARRAY').split(',')

    if args.bootstrap:
        BootStrap(args.host_port)
    else:
        NodeBase(args.port) 
