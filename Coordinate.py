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

    def Split_Axis(self, origin_hash_map, join_hash_map, dimension):
        # Split peer coordinate to join coordinate setting.
        self.oh = origin_hash_map
        self.jh = join_hash_map
        self.dimension = dimension
        self.cut = self.centers
        fail = True
        self.hash_max_distance = 0
        self.max_axis = 0
        
        if self.oh[:-1] == self.jh[:-1]:
            self.max_axis = self.dimension
        
        else :
            for axis in range(self.dimension - 1):
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

                coord1[self.max_axis][1] = int(self.cut[self.max_axis])
                coord2[self.max_axis][0] = int(self.cut[self.max_axis])
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
