#!/usr/bin/env python3
# coding: utf-8
# 进行node的staus更新,查询并给出feature vector

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

def update():
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

#定义匹配结果
#define the query result
def find_image_quality(data0):
    image_quality = ''
    if data0 == 0:
        image_quality = 'no_bright_star image'
    elif data0 == 1:
         print('telescope is changing survey area')
    elif data0 == 2:
        image_quality = 'Stick_like image'
    elif data0 == 3:
        image_quality = 'Donut_like image'
    elif data0 == 4:
        image_quality = 'Two_point_like image'
    elif data0 == 5:
        print('telescope is normal')
    else:
        image_quality = 'Lumpy_like image'
    return image_quality

def merge_feature(core_phase):
    graph = Graph('http://localhost:7474', user='neo4j', password='19830722')
    a0 = find_image_quality(core_phase)
    # 提取参数特征并转化为特征向量
    # Extract the node parameter and merge to feature vectors
    b = graph.run("MATCH (n{classes:'enviroment'})"
                  "MATCH p=({name:n.name})-[*]->({name:$name}) RETURN nodes(p)",name = a0)
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
                  "MATCH p=({name:n.name})-[*]->({name:$name}) RETURN nodes(p)",name = a0)
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
    return data
