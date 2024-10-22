import networkx as nx
import matplotlib.pyplot as plt
from models import build_tree

def visualize_tree(scene):
    root = build_tree(scene)
    G = nx.Graph()

    def add_edges(node):
        for child in node.children:
            G.add_edge(node.name, child.name)
            add_edges(child)

    add_edges(root)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color='lightblue', font_size=10, font_weight='bold')
    plt.title(f"Tree Structure for {scene} Scene")
    plt.show()
