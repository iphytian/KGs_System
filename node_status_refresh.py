#!/usr/bin/env python3
# coding: utf-8
# 进行node的staus更新

from py2neo import Graph,Node,Relationship,NodeMatcher
import time

# 使用nodematcher找到节点
# Use nodematcher to find nodes
def findNode(name, graph):
    matcher = NodeMatcher(graph)
    m = matcher.match(name = name).first()
    return m

def read_csv(file_name):
    f = open(file_name, 'r')
    content = f.read()
    final_list = list()
    rows = content.split('\n')
    i = 0
    for row in rows:
        if i == 0:
            i = i + 1
            continue
        else:
            final_list.append(row.split(','))
    return final_list

graph = Graph('http://localhost:7474',user='neo4j',password='19830722')
time_start = time.time()
data = read_csv('C:/Users/iphy/Desktop/updata_parameter.csv')

for i in range(len(data)):
    match_data = findNode(data[i][0], graph)
    if match_data ==None:
        break
    else:
        match_data.update({'status1': data[i][1]})
        match_data.update({'status2': data[i][2]})
        graph.push(match_data) # push()方法将更新后的节点压入图中 The push() method pushes the updated node into the graph

# 提取参数特征并转化为特征向量
# Extract the node parameter and merge to feature vectors
b = graph.run("MATCH (n{classes:'enviroment'})"
              "MATCH p=({name:n.name})-[*]->({name:$name}) RETURN nodes(p)",name = 'Stick_like image')
new = b.data()
data = []
for i in range(len(new)):
    new1 = new[i]['nodes(p)']
    data1 = []
    print(i)
    print(new1)
    for j in range(len(new1)):
        data1.append(new1[j]['status1'])
        data1.append(new1[j]['status2'])
    data.append(data1)

c = graph.run("MATCH (n{classes:'device'})"
              "MATCH p=({name:n.name})-[*]->({name:$name}) RETURN nodes(p)",name = 'Stick_like image')
new = c.data()
print(new)
for i in range(len(new)):
    new1 = new[i]["nodes(p)"]
    print(i)
    print(new1)
    data1 = []
    for j in range(len(new1)):
        data1.append(new1[j]['status1'])
        data1.append(new1[j]['status2'])
    data.append(data1)
print(data)

time_end = time.time()
time_sum = time_end - time_start
print(time_sum)
