#!/usr/bin/env python3
# coding: utf-8
# 进行node的staus更新

from py2neo import Graph,Node,Relationship,NodeMatcher

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

data = read_csv('C:/Users/iphy/Desktop/updata_parameter.csv')

for i in range(len(data)):
    match_data = findNode(data[i][0], graph)
    if match_data ==None:
        break
    else:
        match_data.update({'status1': data[i][1]})
        match_data.update({'status2': data[i][2]})
        graph.push(match_data) # push()方法将更新后的节点压入图中 The push() method pushes the updated node into the graph