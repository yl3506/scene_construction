# models.py

import numpy as np
from collections import deque

class Node:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children else []

def build_tree(scene):
    if scene == 'Drop':
        root = Node('Drop')
        loc = Node('loc(s1,s2)')
        size = Node('size(s1)')
        traj = Node('traj(s1)')
        person = Node('person')
        type_s1 = Node('type(s1)')
        type_s2 = Node('type(s2)')
        color_s1 = Node('color(s1)')
        color_s2 = Node('color(s2)')
        material_s1 = Node('material(s1)')
        root.children = [type_s1, type_s2, person]
        type_s1.children = [color_s1, size, traj]
        color_s1.children = [material_s1]
        type_s2.children = [color_s2]
    elif scene == 'Push':
        root = Node('Push')
        loc = Node('loc(s1,s2)')
        weight = Node('weight(s1)')
        direction = Node('direction(s1)')
        surface = Node('surface')
        type_s1 = Node('type(s1)')
        type_s2 = Node('type(s2)')
        color_s1 = Node('color(s1)')
        color_s2 = Node('color(s2)')
        material_s1 = Node('material(s1)')
        root.children = [type_s1, type_s2, surface]
        type_s1.children = [color_s1, direction, weight]
        color_s1.children = [material_s1]
        type_s2.children = [color_s2]
    elif scene == 'Pull':
        root = Node('Pull')
        loc = Node('loc(s1,s2)')
        size = Node('size(s1)')
        traj = Node('traj(s1)')
        person = Node('person')
        type_s1 = Node('type(s1)')
        type_s2 = Node('type(s2)')
        color_s1 = Node('color(s1)')
        color_s2 = Node('color(s2)')
        material_s1 = Node('material(s1)')
        root.children = [type_s1, type_s2, person]
        type_s1.children = [color_s1, size, traj]
        color_s1.children = [material_s1]
        type_s2.children = [color_s2]
    else:
        raise ValueError(f"Unknown scene: {scene}")
    return root

def deterministic_bfs(root, effective_time_limit):
    queue = deque()
    queue.append(root)
    visited = []
    time_spent = 0

    while queue and time_spent < effective_time_limit:
        node = queue.popleft()
        visited.append(node.name)
        time_spent += 1
        for child in node.children:
            queue.append(child)
    return visited

def deterministic_dfs(node, effective_time_limit, visited=None, time_spent=0):
    if visited is None:
        visited = []
    if time_spent >= effective_time_limit:
        return visited, time_spent

    visited.append(node.name)
    time_spent += 1

    for child in node.children:
        if time_spent >= effective_time_limit:
            break
        visited, time_spent = deterministic_dfs(child, effective_time_limit, visited, time_spent)
    return visited, time_spent

def probabilistic_bfs(root, effective_time_limit, expansion_prob):
    queue = deque()
    queue.append(root)
    visited = []
    time_spent = 0

    while queue and time_spent < effective_time_limit:
        node = queue.popleft()
        visited.append(node.name)
        time_spent += 1

        for child in node.children:
            if np.random.rand() < expansion_prob:
                queue.append(child)
    return visited

def probabilistic_dfs(node, effective_time_limit, expansion_prob, visited=None, time_spent=0):
    if visited is None:
        visited = []
    if time_spent >= effective_time_limit:
        return visited, time_spent

    visited.append(node.name)
    time_spent += 1

    for child in node.children:
        if time_spent >= effective_time_limit:
            break
        if np.random.rand() < expansion_prob:
            visited, time_spent = probabilistic_dfs(child, effective_time_limit, expansion_prob, visited, time_spent)
    return visited, time_spent
