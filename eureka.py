""" Created on 2023
@author: Yuchan Lee (Korea Aerospace Univ)
"""
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
import hashlib
import copy
import sys
import os
import re
from matplotlib import pyplot as plt

class Eureka:
    def __init__(self):
        self.dim = args.dimension
        self.axis_max = args.max_coordinate
        self.port = args.port
        self.sucess = False
        self.nodes = args.node_nums
        self.axis_max_len = len(str(self.axis_max))

        self.table = []
        self.error_count = 0
        self.index = 0
        self.port = 12000

        for i in range(len(server_list)):
            #print('send to data', args.host_addr, 12000 +i)
            for j in range(int(args.node_nums/len(server_list))):
                node_addr = (server_list[self.index], self.port)
                #print(node_addr)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(node_addr)
                data = ('node scan', node_addr, this_addr)
                sock.send(pickle.dumps(data))
                sock.close()
                self.port += 1
                time.sleep(0.1)
            self.index += 1
        #time.sleep(0.1 * self.nodes + 10)
        log_file = open("/home/deepl/CAN_python/log.txt", "w")
        log_file.write("Verification start!")
        log_file.close()
        print("Verification start!")
        time.sleep(30)

        for node in range(self.nodes):   ##append node's information to the table
            node_file="Node%d.txt"%(node)
            file_name=args.route+"/"+node_file
            f = open(file_name,"r")
            coord = [[]for _ in range(self.dim)]
            hashmap = []
            neigh=[]
            di={}

            for i in range(3):
                line = f.readline()
                redex = re.compile("\d{0,%d}\.?\d+"%(self.axis_max_len))
                str_numbers=redex.findall(line)
                numbers = list(map(float,str_numbers))
                block = 1 + self.dim + 2 * self.dim

                if (i==0):	##append own coordination
                    for j in range(len(numbers)):
                        coord[j//2].append(numbers[j])
                elif(i==1):	##append own hashmap
                    for j in range(len(numbers)):
                        hashmap.append(numbers[j])
                elif(i==2):
                    for j in range(len(numbers)):   ##append own neighbor node's number
                        if(j%block==0):
                            neigh.append(int(numbers[j]))
                            if(len(neigh)>=self.nodes):	##notice if neighbor table has neighbor node number greater than the largest node number
                                print("total nodes: %d  but Node%d has Node%d as a neighbor"%(self.nodes,node,int(numbers[j])))

            di['node']= node 
            di['hashmap'] = hashmap
            if(len(hashmap)!=self.dim):		##notice if the hashmap information is different from dimension information
                print("Node%d's hashmap information is different from dimension  Node%d's hashmap dimension: %d  provided dimension: %d"%(node,node,len(hashmap),self.dim))
            di['coord'] = coord  
            if(len(coord)!=self.dim):
                print("Node%d's coordinate informatino is different from dimensino  Node%d's coordinate dimension: %d  provided dimension: %d"%(node,node,len(coord),self.dim))
            di['neighbor'] = neigh
            self.table.append(di)

        for node in range(self.nodes):	## get neighbor information fromm neighbor table
            neigh_table = []
            neigh_coord = [[]for _ in range(self.dim)]
            neigh_hashmap = []
            node_file="Node%d.txt"%(node)
            file_name=args.route+"/"+node_file
            f = open(file_name,"r")
            for i in range(3):
                line = f.readline()
            redex = re.compile("\d{0,%d}\.?\d+"%(self.axis_max_len))
            str_numbers=redex.findall(line)
            numbers = list(map(float,str_numbers))
            block = 1 + self.dim + 2 * self.dim
            neigh_count = len(numbers) // block
            for j in range(len(numbers)):   
                if(j%block==0):	## initialize neighbor information
                    neigh_di = {}
                    neigh_di['node']=int(numbers[j])	## append neighbor number to dictionary
                    neigh_coord = [[]for _ in range(self.dim)]
                    neigh_hashmap = []
                elif(1<=j%block<=self.dim):	## append neighbor hashmap to neigh_hashmap list
                    neigh_hashmap.append(numbers[j])
                elif(self.dim<j%block<block-1):	## append neighbor coordinate to neigh_coord list
                    neigh_coord[((j%block)-self.dim-1)//2].append(numbers[j])
                elif(j%block==block-1):
                    neigh_coord[((j%block)-self.dim-1)//2].append(numbers[j])
                    neigh_di['hashmap']= neigh_hashmap	## add neigh_hashamp to neighbor dictionaty
                    neigh_di['coord'] = neigh_coord	##add neigh_coord to neighbor dictionary
                    neigh_table.append(neigh_di)	## append neighbor dictionary to neigh_table

            for j in range (len(neigh_table)):	## verify information in the neighbor table matches the information in the table
                node_num=neigh_table[j].get('node')
                if(self.table[node_num].get('hashmap')!=neigh_table[j].get('hashmap')):
                    print('\n',self.table[node_num].get('hashmap'),'  table hash map')
                    print(neigh_table[j].get('hashmap'),'  neighbor table hash map')
                    print("Node%d has different hashmap of Node%d"%(node,node_num))
                    self.error_count+=1
                if(self.table[node_num].get('coord')!=neigh_table[j].get('coord')):
                    print('\n',self.table[node_num].get('coord'),'  table coordinate')
                    print(neigh_table[j].get('coord'),'  neighbor table coordinate')
                    print("Node%d has different coordinate of Node%d"%(node,node_num))
                    self.error_count+=1

        self.neighbor_check()
        self.position()
        self.entire_space()
        self.eureka()
        self.coord_stats()
        self.neigh_stats()

    def neighbor_check(self):	## check my neighbor thinks me as a neighbor

        for i in range(self.nodes):
            my_neighbor = self.table[i].get('neighbor')

            for j in range(len(my_neighbor)):
                self.index = my_neighbor[j]          
                your_neighbor = self.table[self.index].get('neighbor')

            if( i in your_neighbor):
                break
            else:
                print("\nNode%d did not register Node%d as a neighbor"%(self.index,i))
                self.error_count+=1
        print("neighbor check done")

    def position(self):	##check hashmap is located inside the coordinates

        for i in range(self.nodes):  
            hashmap = self.table[i].get('hashmap')
            coord = self.table[i].get('coord')

        for j in range(self.dim):
            if(coord[j][0]<=hashmap[j]<coord[j][1]):
                pass
            else:
                self.error_count +=1
                print("\nNode%d's hashmap shared_axis%d is out of boundary"%(i,j))
        print("position check done")

    def entire_space(self): ##check that the sum of the nodes matches the size of the entire space
        space =0
        real_space = self.axis_max**self.dim

        for i in range(self.nodes):
            cal_space = 1
            node_coord = self.table[i].get('coord')

        for j in range(self.dim):
            cal_space *= (node_coord[j][1]-node_coord[j][0])

        space += cal_space

        if(space!=real_space):
            print('\nreal_space=',real_space)
            print('space=',space)
            print("there is empty space")

        print('\nreal_space=',real_space)
        print('space=',space)
        print('entire space check done')

    def eureka(self):

        for i in range(self.nodes):
            my_coord = self.table[i].get('coord')
            my_neighbor = self.table[i].get('neighbor')
            axis_space =[0]*self.dim
            touch_space = [0]*self.dim
            my_space = 1

            for k in range(self.dim):        ## total size of node
                my_space *= (my_coord[k][1]-my_coord[k][0])
            for k in range(self.dim):        ## size of outer shell of a node
                if(my_coord[k][0]==0 and my_coord[k][1]==self.axis_max):	## the lower and upper of the coordinate are both in contact with the boundary
                    axis_space[k] = 0
                elif(my_coord[k][0]==0 or my_coord[k][1]==self.axis_max):	## the lower or upper of the coordinate touches the boundary
                    axis_space[k] = my_space/(my_coord[k][1]-my_coord[k][0])
                else:								## the lower and upper of the coordinate are not at the boundary
                    axis_space[k] = my_space*2/(my_coord[k][1]-my_coord[k][0])

            for j in range(len(my_neighbor)):
                self.index = my_neighbor[j]       
                neigh_coord = self.table[self.index].get('coord')  
                for shared_axis in range(self.dim):    ## which axis is orthogonal to the node
                    if(my_coord[shared_axis][0] == neigh_coord[shared_axis][1] or my_coord[shared_axis][1]==neigh_coord[shared_axis][0]):
                        break 
                    elif(shared_axis==(self.dim-1)):	## Node and neighbor node are not touch to each other
                        self.error_count +=1
                        print('\nneigh: ',neigh_coord)
                        print('node: ', my_coord)
                        print("Node%dis out of touch with Node%d"%(self.index,i))
                shared_coord = 1

                for axis in range(self.dim):
                    length = 0
                    if (axis==shared_axis): ## not calculating the orthogonal axis
                        continue

                    if(my_coord[axis][0]>neigh_coord[axis][0]):    ## lower중 더 큰 쪽이 접한 공간의 lower
                        touch_low= my_coord[axis][0]
                    else:
                        touch_low=neigh_coord[axis][0]

                    if(my_coord[axis][1]<neigh_coord[axis][1]):  ##upper중 더 작은 쪽이 접한 공간의 upper
                        touch_up= my_coord[axis][1]
                    else:
                        touch_up=neigh_coord[axis][1]

                    if(touch_low >=touch_up):
                        print('\nneigh: ',neigh_coord)
                        print('node: ', my_coord)
                        print('up: %d low:%d'%(touch_up,touch_low))
                        print('length: ', touch_up-touch_low)
                        print("check Node%d is touch with Node%d"%(i,self.index))
                        continue
                    length=touch_up-touch_low
                    shared_coord *= length

                touch_space[shared_axis]+=shared_coord

            for j in range(self.dim):
                if(touch_space[j]<axis_space[j]):
                    print("\nneighbor touch space: ",touch_space[j])
                    print("real boundary space: ", axis_space[j])
                    print("Node%d's axis%d has smaller touch space than real boundary space\n"%(i,j),my_neighbor)

                elif(touch_space[j]>axis_space[j]):
                    for n1 in range(len(my_neighbor)-1):
                        n1_index=my_neighbor[n1]
                        n1_coord=self.table[n1_index].get('coord')

                    for n2 in range(n1+1,len(my_neighbor)):
                        n2_index = my_neighbor[n2]
                        n2_coord=self.table[n2_index].get('coord')
                        axis_check=0

                    for axis in range(self.dim):
                        if(n1_coord[axis][0]>n2_coord[axis][0]):
                            check_low= n1_coord[axis][0]
                        else:
                            check_low=n2_coord[axis][0]

                        if(n1_coord[axis][1]<n2_coord[axis][1]):
                            check_up=n1_coord[axis][1]
                        else:
                            check_up=n2_coord[axis][1]

                        if(check_low<check_up):
                            axis_check+=1

                    if(axis_check>=self.dim):
                        print("\nNode%d is overlapped with Node%d"%(n1_index,n2_index))
        print("eureka done")

    def neigh_stats(self):	##distribution of how many neighbors each node has
        stats=[]
        for node in range(self.nodes):
            num_neigh=len(self.table[node].get('neighbor'))
            stats.append(num_neigh)
        plt.hist(stats,bins=50)
        plt.title('the distribution of neighbors')
        plt.xlabel('the number of neighbors')
        plt.ylabel('the number of nodes')
        plt.savefig('neigh.png')
        plt.close()

    def coord_stats(self):	## size distribution of nodes
        stats=[]
        for node in range(self.nodes):
            coord = self.table[node].get('coord')
            size=1
            for i in range(self.dim):
                length = coord[i][1]-coord[i][0]
                size *= length
            stats.append(size)
        plt.hist(stats,bins=100)
        plt.xscale('log')
        plt.title('the distribution of size')
        plt.xlabel('size of node')
        plt.ylabel('the number of nodes')
        plt.savefig('size.png')
        plt.close()

if __name__ == '__main__':

    this_ip = socket.gethostbyname(socket.gethostname())
    parser = argparse.ArgumentParser(description="eureka")
    parser.add_argument("--port","-p", help="verificaion's port number", type=int, default=18000)
    parser.add_argument("--host_port","-P", help="help bootstrap's port number", type=int, default=12000)
    parser.add_argument("--host_addr","-A", help="help peer's ip address", type=str, default='220.67.133.165')
    parser.add_argument('-d','--dimension',type=int,default= 5)
    parser.add_argument('-m','--max_coordinate',type=float,default=65536)
    parser.add_argument('-n','--node_nums',type=int,default=100)
    parser.add_argument('-r','--route',default="log")
    args=parser.parse_args()

    server_list=[]
    this_addr = (this_ip, args.port)
    host_addr = (args.host_addr, args.host_port)

    print("Verification Node IP is:", this_ip)
    print("Verification Node Port is:", args.port)

    server_list = os.environ.get('SERVER_ARRAY').split(',')
    print('Verification server list :', server_list)

    print('Eureka Start!')
    Eureka()
