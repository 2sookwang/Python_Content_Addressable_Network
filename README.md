# Python Content Addressable Network(CAN)
Content Addressable Network (CAN) code implemented in python

## 🖥️ Project introduce
We are working on a project that implements DHT (Distributed hash tables) CAN in Python and applies Federated Learning
<br>

## 🕰️ Develop period
* 23.06.12 ~ present

### 🧑‍🤝‍🧑 Participating workforce
 - Master's research student  : Sookwang Lee - Comprehensive CAN code configuration, applied Federated Learning
 - undergraduate research student : Yuchan Lee - CAN node verification code (eureka), CAN code update

### ⚙️ Development environment
- `Python 3.7.4`
- `Shell script`
- **OS** : Ubuntu
- **Framework** : PyTorch

## 📌 Main function
### 1. CAN node join
- Running Main.py
### How to run
**parameter value**
- `port` : this peer's port number
- `host_port` : Bootstrap's port number
- `host_addr` : Bootstrap's ip address
- `dimension` : node dimension setting
- `bootstrap` : is Bootstrap?(True/False)
- `node_num` : number of node
- `node_nums` : total number of node
- `max_coordinate` : Coordinate max value
- `hash_text` : hash text

### Running Example 
**Experiment setup :** `7 dimension` `2048 Node join` `Coordinate size 65536`

**Bootstrap**
```
python3 Main.py --host_addr=220.63.132.101 --bootstrap=True --max_coordinate=65536 --dimension=7 --port=12000 --node_num=0 --node_nums=2048
```
**Node**
```
python3 Main.py --host_addr=220.63.132.102 --bootstrap=True --max_coordinate=65536 --dimension=7 --port=12001 --node_num=0 --node_nums=2048`
```
## 
### 2. Verification CAN Coordinate
- Running eureka.py
### How to run
**parameter value**
- `port` : this verification's port number
- `host_port` : Bootstrap's port number
- `host_addr` : Bootstrap's ip address
- `dimension` : node dimension setting
- `max_coordinate` : Coordinate max value
- `node_nums` : total number of node
- `route` : Node.txt file location for verification

### Running Example 
**Experiment setup :** `7 dimension` `2048 Node join` `Coordinate size 65536`
```
python3 eureka.py --node_nums=2048 --dimension=7 --max_coordinate=65536 --route=log0812
```
## 

### 3. Running in Multi node environment
- Running shellscript.sh
#### How to run
**parameter value** : mode_number dimension max_coordinate total_node_nums number_of_server server_ip1 server_ip2 ...
- `mode number` : 1(Bootstrap + Nodes), 2(Nodes), 3(verification code + make log file)
- `dimension` : CAN code dimension setting
- `max_coordinate` : CAN code max cooordinate setting
- `total node nums` : input node scale (512, 1024, 2048 ... etc)
- `number of server` : number of servers used (Number of joins within a node = total number of nodes / number of servers used)
- `server_ip` : ip of the server being used

### Running Example 
**Experiment setup :** `7 dimension` `Coordinate size 65536` `2048 Node join`  `number of server 4` 
```
./shellscript.sh 1 7 65536 2048 0 4 '220.63.132.101' '220.63.132.102' '220.63.132.72' 220.63.131.150'
```
