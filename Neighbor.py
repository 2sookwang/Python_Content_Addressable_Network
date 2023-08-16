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
