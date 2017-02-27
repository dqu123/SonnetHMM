import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import numpy as np

mat = np.matrix([[0.13094, 0.0, 0.0, 0.0, 0.24661, 0.0, 0.0, 0.62245, 0.0, 0.0], [0.4074, 0.0, 0.1259, 0.28116, 0.0, 0.0, 0.0, 0.0, 0.18554, 0.0], [0.07904, 0.0, 0.00766, 0.40468, 0.04543, 0.0, 0.0, 0.0, 0.46319, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03553, 0.0, 0.96447], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.96102, 0.0, 0.0, 0.03898, 0.0, 0.0, 0.0, 0.0], [0.0, 0.07161, 0.02406, 0.07865, 0.25297, 0.012, 0.0, 0.0, 0.03941, 0.5213], [0.60519, 0.0, 0.04848, 0.08196, 0.0, 0.01917, 0.0, 0.0, 0.2452, 0.0], [0.08864, 0.86739, 0.0, 0.0, 0.0, 0.00104, 0.0, 0.0, 0.04293, 0.0], [0.0, 0.0, 0.0, 0.0, 0.46889, 0.0, 0.0, 0.00249, 0.44196, 0.08665]])

G = nx.DiGraph(mat)
pos = nx.spring_layout(G)
labels = nx.get_edge_attributes(G, 'weight')

mymap = plt.get_cmap("Blues")
colors = np.linspace(0.1, 1, 5)
my_colors = mymap(colors)

edgelists = [[] for _ in range(5)]
for (u, v, d) in G.edges(data=True):
    for i in range(5):
        if d['weight'] <= (i + 1) * 1. / 5:
            edgelists[i].append((u,v))
            break

nx.draw_networkx_nodes(G, pos, node_color='b')
nx.draw_networkx_labels(G, pos)
handles = []
for i in range(5):
    color = matplotlib.colors.to_hex(my_colors[i])
    nx.draw_networkx_edges(G, pos, edgelist=edgelists[i], arrows=True, edge_color=color)
    handles.append(mpatches.Patch(color=color, label='{} - {}'.format(i * 1. / 5, (i+1) * 1. / 5)))

plt.legend(handles=handles)

plt.show()

