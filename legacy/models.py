import numpy as np
from collections import deque

class Node:
    def __init__(self, name, is_dummy=False, children=None):
        self.name = name
        self.children = children if children else []
        self.is_dummy = is_dummy

def build_tree(scene):
    if scene == 'Drop':
        root = Node('Drop')
        s1 = Node('s1', is_dummy=True)
        s2 = Node('s2', is_dummy=True)
        person = Node('person')
        surface = Node('surface', is_dummy=True)
        what_s1 = Node('what(s1)', is_dummy=True)
        what_s2 = Node('what(s2)', is_dummy=True)
        where_s1 = Node('where(s1)', is_dummy=True)
        where_s2 = Node('where(s2)', is_dummy=True)
        how_s1 = Node('how(s1)', is_dummy=True)
        how_s2 = Node('how(s2)', is_dummy=True)
        type_s1 = Node('type(s1)')
        type_s2 = Node('type(s2)')
        loc_s1 = Node('loc(s1, s2)')
        loc_s2 = Node('loc(s1, s2)', is_dummy=True)
        direction_s1 = Node('direction(s1)', is_dummy=True)
        direction_s2 = Node('direction(s2)', is_dummy=True)
        color_s1 = Node('color(s1)')
        color_s2 = Node('color(s2)')
        size_s1 = Node('size(s1)')
        size_s2 = Node('size(s2)', is_dummy=True)
        traj_s1 = Node('traj(s1)')
        traj_s2 = Node('traj(s2)', is_dummy=True)
        texture_s1 = Node('texture(s1)', is_dummy=True)
        texture_s2 = Node('texture(s2)', is_dummy=True)
        weight_s1 = Node('weight(s1)', is_dummy=True)
        weight_s2 = Node('weight(s2)', is_dummy=True)
        material_s1 = Node('material(s1)')
        material_s2 = Node('material(s2)', is_dummy=True)
        root.children = [s1, s2, person, surface]
        s1.children = [what_s1, where_s1, how_s1]
        s2.children = [what_s2, where_s2, how_s2]
        what_s1.children = [type_s1]
        what_s2.children = [type_s2]
        where_s1.children = [loc_s1]
        where_s2.children = [loc_s2]
        how_s1.children = [direction_s1]
        how_s2.children = [direction_s2]
        type_s1.children = [color_s1]
        type_s2.children = [color_s2]
        loc_s1.children = [size_s1]
        loc_s2.children = [size_s2]
        direction_s1.children = [traj_s1]
        direction_s2.children = [traj_s2]
        color_s1.children = [texture_s1]
        color_s1.children = [texture_s2]
        size_s1.children = [weight_s1]
        size_s2.children = [weight_s2]
        texture_s1.children = [material_s1]
        texture_s2.children = [material_s2]
    elif scene == 'Push':
        root = Node('Push')
        s1 = Node('s1', is_dummy=True)
        s2 = Node('s2', is_dummy=True)
        person = Node('person', is_dummy=True)
        surface = Node('surface')
        what_s1 = Node('what(s1)', is_dummy=True)
        what_s2 = Node('what(s2)', is_dummy=True)
        where_s1 = Node('where(s1)', is_dummy=True)
        where_s2 = Node('where(s2)', is_dummy=True)
        how_s1 = Node('how(s1)', is_dummy=True)
        how_s2 = Node('how(s2)', is_dummy=True)
        type_s1 = Node('type(s1)')
        type_s2 = Node('type(s2)')
        loc_s1 = Node('loc(s1, s2)')
        loc_s2 = Node('loc(s1, s2)', is_dummy=True)
        direction_s1 = Node('direction(s1)')
        direction_s2 = Node('direction(s2)', is_dummy=True)
        color_s1 = Node('color(s1)')
        color_s2 = Node('color(s2)')
        size_s1 = Node('size(s1)', is_dummy=True)
        size_s2 = Node('size(s2)', is_dummy=True)
        traj_s1 = Node('traj(s1)', is_dummy=True)
        traj_s2 = Node('traj(s2)', is_dummy=True)
        texture_s1 = Node('texture(s1)', is_dummy=True)
        texture_s2 = Node('texture(s2)', is_dummy=True)
        weight_s1 = Node('weight(s1)')
        weight_s2 = Node('weight(s2)', is_dummy=True)
        material_s1 = Node('material(s1)')
        material_s2 = Node('material(s2)', is_dummy=True)
        root.children = [s1, s2, person, surface]
        s1.children = [what_s1, where_s1, how_s1]
        s2.children = [what_s2, where_s2, how_s2]
        what_s1.children = [type_s1]
        what_s2.children = [type_s2]
        where_s1.children = [loc_s1]
        where_s2.children = [loc_s2]
        how_s1.children = [direction_s1]
        how_s2.children = [direction_s2]
        type_s1.children = [color_s1]
        type_s2.children = [color_s2]
        loc_s1.children = [size_s1]
        loc_s2.children = [size_s2]
        direction_s1.children = [traj_s1]
        direction_s2.children = [traj_s2]
        color_s1.children = [texture_s1]
        color_s1.children = [texture_s2]
        size_s1.children = [weight_s1]
        size_s2.children = [weight_s2]
        texture_s1.children = [material_s1]
        texture_s2.children = [material_s2]
    elif scene == 'Pull':
        root = Node('Pull')
        s1 = Node('s1', is_dummy=True)
        s2 = Node('s2', is_dummy=True)
        person = Node('person')
        surface = Node('surface', is_dummy=True)
        what_s1 = Node('what(s1)', is_dummy=True)
        what_s2 = Node('what(s2)', is_dummy=True)
        where_s1 = Node('where(s1)', is_dummy=True)
        where_s2 = Node('where(s2)', is_dummy=True)
        how_s1 = Node('how(s1)', is_dummy=True)
        how_s2 = Node('how(s2)', is_dummy=True)
        type_s1 = Node('type(s1)')
        type_s2 = Node('type(s2)')
        loc_s1 = Node('loc(s1, s2)')
        loc_s2 = Node('loc(s1, s2)', is_dummy=True)
        direction_s1 = Node('direction(s1)', is_dummy=True)
        direction_s2 = Node('direction(s2)', is_dummy=True)
        color_s1 = Node('color(s1)')
        color_s2 = Node('color(s2)')
        size_s1 = Node('size(s1)')
        size_s2 = Node('size(s2)', is_dummy=True)
        traj_s1 = Node('traj(s1)')
        traj_s2 = Node('traj(s2)', is_dummy=True)
        texture_s1 = Node('texture(s1)', is_dummy=True)
        texture_s2 = Node('texture(s2)', is_dummy=True)
        weight_s1 = Node('weight(s1)', is_dummy=True)
        weight_s2 = Node('weight(s2)', is_dummy=True)
        material_s1 = Node('material(s1)')
        material_s2 = Node('material(s2)', is_dummy=True)
        root.children = [s1, s2, person, surface]
        s1.children = [what_s1, where_s1, how_s1]
        s2.children = [what_s2, where_s2, how_s2]
        what_s1.children = [type_s1]
        what_s2.children = [type_s2]
        where_s1.children = [loc_s1]
        where_s2.children = [loc_s2]
        how_s1.children = [direction_s1]
        how_s2.children = [direction_s2]
        type_s1.children = [color_s1]
        type_s2.children = [color_s2]
        loc_s1.children = [size_s1]
        loc_s2.children = [size_s2]
        direction_s1.children = [traj_s1]
        direction_s2.children = [traj_s2]
        color_s1.children = [texture_s1]
        color_s1.children = [texture_s2]
        size_s1.children = [weight_s1]
        size_s2.children = [weight_s2]
        texture_s1.children = [material_s1]
        texture_s2.children = [material_s2]
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
        if not node.is_dummy:
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
    if not node.is_dummy:
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
        if not node.is_dummy:
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
    if not node.is_dummy:
        visited.append(node.name)
    time_spent += 1
    for child in node.children:
        if time_spent >= effective_time_limit:
            break
        if np.random.rand() < expansion_prob:
            visited, time_spent = probabilistic_dfs(child, effective_time_limit, expansion_prob, visited, time_spent)
    return visited, time_spent



