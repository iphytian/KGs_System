#!/usr/bin/env python3
# coding: utf-8
import csv
from py2neo import Graph,Node,Relationship,NodeMatcher
# 启动知识图谱
# Start the Knowledge Graph
g = Graph('http://localhost:7474',user='neo4j',password='19830722')
nodes = NodeMatcher(g)

# 导入数据，构造知识图谱
# Import data and construct knowledge graph
# 构造节点
# construct node
with open('C:\\Users\\iphy\\Desktop\\node.csv','r',encoding='utf-8') as f:
    reader=csv.reader(f)
    for item in reader:
        if reader.line_num==1:
            continue
        print("当前行数：",reader.line_num,"当前内容：",item)
        node=Node(item[2],classes=item[1],id=item[0],name=item[3],status1=item[4],status2=item[5])
        g.create(node)