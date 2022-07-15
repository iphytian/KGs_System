#!/usr/bin/env python3
# coding: utf-8
import csv
from py2neo import Graph,Node,Relationship,NodeMatcher
g = Graph('http://localhost:7474',user='neo4j',password='19830722')
nodes = NodeMatcher(g)
print(nodes)
# 构造节点之间的关系
# Construct the relationship between nodes
with open('C:\\Users\\iphy\\Desktop\\edge.csv','r',encoding='utf-8') as f:
    reader=csv.reader(f)
    for item in reader:
        if reader.line_num == 1:
            continue
        print("当前行数：",reader.line_num,"当前内容：",item)
        a1 = int(item[0])
        a2 = int(item[1])
        b1 = nodes.get(a1)
        b2 = nodes.get(a2)
        node = Relationship(b1,item[2],b2)
        g.create(node)