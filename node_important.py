#!/usr/bin/env python3
# coding: utf-8
#计算各个节点的重要性
from py2neo import Graph,Node,Relationship,NodeMatcher
graph = Graph('http://localhost:7474',user='neo4j',password='19830722')

# 对集合中name相同的list进行合并
# Merge lists with the same name in the set
def merge_lsit(set):
    set0 = []
    for i in range(len(set)):
        score = set[i][0]
        name = set[i][1]
        for j in range(len(set)):
            a = set[i][1]
            b = set[j][1]
            if (i != j) and (a == b):
                score = score +set[j][0]
            else:
                continue
        val = 0
        for k in range(len(set0)):
            if len(set0) > 0:
                if set0[k][1] == name:
                    val = val + 1
                    break
                else:
                    val = val + 0
            else:
                val = 0
        if val == 0:
            set0.append([score,name])
        else:
            continue
    return set0

# 在集合中找到某个属性的最大值
# Find the maximum value of an attribute in the collection
def find_max(set):
    name = set[0][1]
    score = set[0][0]
    for i in range(len(set)):
        if set[i][0]>score:
            score = set[i][0]
            name = set[i][1]
        else:
            continue
    return name, score

# 定义空列表
# define empty list
score = 0
# device_sets = []
# # 所有装置引起故障，节点重要性计算
# # All devices cause failure, node importance calculation
# b = graph.run("MATCH (n{classes:'device'})"
#               "MATCH p=({name:n.name})-[*]->({classes:'image_quality'}) RETURN n.name, p,length(p)")
#
# while b.forward():
#     rel = b.current["p"]
#     num = int(b.current["length(p)"])
#     score = (1/1000)**num
#     name = b.current["n.name"]
#     device_sets.append([score,name])
#
# for i in range(len(device_sets)):
#     print(device_sets[i])



# 所有环境引起的故障，节点重要性计算
# Failures caused by all environments, node importance calculation
envi_sets = []
b = graph.run("MATCH (n{classes:'enviroment'})"
              "MATCH p=({name:n.name})-[*]->({classes:'image_quality'}) RETURN n.name, p,length(p)")

while b.forward():
    rel = b.current["p"]
    num = int(b.current["length(p)"])
    score = (1/1000)**num
    name = b.current["n.name"]
    envi_sets.append([score,name])

for i in range(len(envi_sets)):
    print(envi_sets[i])

list1 = merge_lsit(envi_sets)
print(list1)
a = find_max(list1)
print(a[0])
print(a[1])
#
# # 所有工作人员引起的故障，节点重要性计算
# # Failure caused by all staff, node importance calculation
# staff_sets = []
# b = graph.run("MATCH (n{classes:'staff'})"
#               "MATCH p=({name:n.name})-[*]->({classes:'image_quality'}) RETURN n.name, p,length(p)")
#
# while b.forward():
#     rel = b.current["p"]
#     num = int(b.current["length(p)"])
#     score = (1/1000)**num
#     name = b.current["n.name"]
#     staff_sets.append([score,name])