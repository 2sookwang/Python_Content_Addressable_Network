#!/bin/bash

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# parameter value : mode_number dimension max_coordinate total_node_nums number_of_server server_ip1 server_ip2 ...
# mode number : 1(Bootstrap + Nodes), 2(Nodes), 3(verification code + make log file)
# dimension : CAN code dimension setting
# max_coordinate : CAN code max cooordinate setting
# total node nums : input node scale (512, 1024, 2048 ... etc)
# number of server : number of servers used (Number of joins within a node = total number of nodes / number of servers used)
# server_ip : ip of the server being used

# ./shellscript.sh 1 7 65536 2048 0 4 5 '220.67.133.165' '220.67.133.166' '220.67.133.82' 220.67.133.110'
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameter initialize
start_time=$(date '+%s')
date=$(date +%m%d)
CURRENTPATH=$( pwd )

M=$1
dimension=$2
max_coordinate=$3
total_node_nums=$4
ip_num=$5
number_of_server=$6

parameter_array=("$@")
server_array=()

node_nums=`expr "$total_node_nums" / "$number_of_server"`
ports=12000
node=`expr "$ip_num" \* "$node_nums"`

for ((i=6; i<${#parameter_array[@]}; i++)); do
    server_array+=("${parameter_array[i]}")
done

echo Parameter initialize done! Mode $M, $dimension dimension, max coordinate $max_coordinate, node scale $total_node_nums, number of server $number_of_server
sleep 1

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mode 1 is initialize Bootstrap and join nodes
if [ $M -eq 1 ];
then 
echo Mode 1 start

pkill -9 python3
source ~/anaconda3/bin/activate fed

# Bootstrap ~ Node(node_nums -1) join
while [ $node -lt $node_nums ]; do
    if [ $ports = '12000' ]
    then 
    (SERVER_ARRAY=$(IFS=,; echo "${server_array[*]}") python3 Main.py --host_addr=${server_array[0]} --bootstrap=True --max_coordinate=$max_coordinate --dimension=$dimension --port=$ports --node_num=$node --node_nums=$total_node_nums) &
    echo Bootstrap initialized!
    ((ports++))
    ((node++))
    sleep 1

    else
    echo Node $node join!
    (SERVER_ARRAY=$(IFS=,; echo "${server_array[*]}") python3 Main.py --host_addr=${server_array[0]} --max_coordinate=$max_coordinate --dimension=$dimension --port=$ports --node_num=$node --node_nums=$total_node_nums) &
    while true; do
        if tail -n 1 log.txt | grep -q "Queue reset!($node)"; then
            ((ports++))
            ((node++))
            break
        fi
        sleep 1
    done
    fi
done
# server_array index++
((ip_num++))

# send to next server(Mode 2) to join Nodes
ssh -p 6304 deepl@${server_array[$ip_num]} "source ~/anaconda3/bin/activate fed; cd Python_Content_Addressable_Network; ./shellscript.sh 2 $dimension $max_coordinate $total_node_nums $ip_num $number_of_server ${server_array[@]}"

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mode 2 is Node join from other servers and last server send Mode 3 to original server 
elif [ $M -eq 2 ];
then
echo Mode 2 start

if [ $ip_num -lt $number_of_server ];
then
pkill -9 python3
ip_node_nums=`expr "$ip_num" \* "$node_nums"` 
ports=`expr "$ports" + "$ip_node_nums"`
node=$ip_node_nums

while [ $node -lt `expr "$node_nums" + "$ip_node_nums"` ]; do
    echo Node $node join!
    SERVER_ARRAY=$(IFS=,; echo "${server_array[*]}") python3 Main.py --host_addr=${server_array[0]} --max_coordinate=$max_coordinate --dimension=$dimension --port=$ports --node_num=$node --node_nums=$total_node_nums &
    while true; do
        if tail -n 1 log.txt | grep -q "Queue reset!($node)"; then
            ((ports++))
            ((node++))
            break
        fi
        sleep 1
    done
done

# server_array index ++ 
((ip_num++))

if [ $ip_num -ne $number_of_server ];
then
# send to next server(Mode 2) to join Nodes
ssh -p 6304 deepl@${server_array[$ip_num]} "source ~/anaconda3/bin/activate fed; cd Python_Content_Addressable_Network; ./shellscript.sh 2 $dimension $max_coordinate $total_node_nums $ip_num $number_of_server ${server_array[@]}"

# if last server join done, send to original server(Mode 3) to verification CAN
elif [ $ip_num -eq $number_of_server ];
then
ssh -p 6304 deepl@${server_array[0]} "source ~/anaconda3/bin/activate fed; cd Python_Content_Addressable_Network; ./shellscript.sh 3 $dimension $max_coordinate $total_node_nums $ip_num $number_of_server ${server_array[@]}"

fi

else
echo Last server join done! Please check your parameters

fi
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Mode 3 is start verification code
elif [ $M = 3 ];
then
SERVER_ARRAY=$(IFS=,; echo "${server_array[*]}") python3 eureka.py --node_nums=$total_node_nums --dimension=$dimension --max_coordinate=$max_coordinate --route=log$date &

while true; do
    if tail -n 1 log.txt | grep -q "Verification start!"; then
        for server in "${server_array[@]}"; do
            echo $server log file upload
            scp -P 6304 deepl@"$server":/home/deepl/CAN_python/log$date/Node*.txt /home/deepl/CAN_python/log$date/
        done
        break
    fi
    sleep 1
done

echo done!
fi

end_time=$(date '+%s')

diff=$((end_time - start_time))
hour=$((diff / 3600 % 24))
minute=$((diff / 60 % 60))
second=$((diff % 60))

echo "$hour hour $minute minute $second second"
