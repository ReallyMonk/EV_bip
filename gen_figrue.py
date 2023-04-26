import pandas as pd
import network2tikz as n2t

# -----------------------------------------------------
# struc2vec(G, 5)
node_id = 'ABCDEFGHIJKLM'
BLUE = (30, 144, 255)
RED = (238, 44, 44)
GREEN = (0, 205, 102)
BLACK = (0, 0, 0)

node_num = 5

nodes = []

# set location
base_location = [(0, 0) for i in range(node_num)]
base_location[1] = (0.7, -0.6)
base_location[2] = (0.4, -1.4)
base_location[3] = (-base_location[1][0], base_location[1][1])
base_location[4] = (-base_location[2][0], base_location[2][1])

g_node = [0, 6]
b_node = [1, 3, 5, 7, 8]
r_node = [2, 4, 9]

# scale the location
node_location = []
for item in base_location:
    node_location.append((item[0] * 1.5 - 2, item[1] * 1.5 - 2))

for i in range(node_num):
    node = {}
    node['id'] = node_id[i]
    node['x'] = node_location[i][0]
    node['y'] = node_location[i][1]

    node['opacity'] = .9
    node['size'] = .3

    if i in b_node:
        node['R'] = BLUE[0]
        node['G'] = BLUE[1]
        node['B'] = BLUE[2]
    elif i in g_node:
        node['R'] = GREEN[0]
        node['G'] = GREEN[1]
        node['B'] = GREEN[2]
    elif i in r_node:
        node['R'] = RED[0]
        node['G'] = RED[1]
        node['B'] = RED[2]
    else:
        node['R'] = 0
        node['G'] = 0
        node['B'] = 0

    nodes.append(node)

nodes = pd.DataFrame(nodes)
nodes.to_csv('./reportdata/2_nodes.csv', index=False)

#--------------------------------------------------
new_nodes_id = [i + 5 for i in range(node_num)]

new_node_location = []
for item in node_location:
    new_node_location.append((item[0], item[1] + 5))

new_nodes = []
for i in range(node_num):
    node = {}
    node['id'] = node_id[new_nodes_id[i]]
    node['x'] = new_node_location[i][0]
    node['y'] = new_node_location[i][1]

    node['opacity'] = .9
    node['size'] = .3

    if i + 5 in b_node:
        node['R'] = BLUE[0]
        node['G'] = BLUE[1]
        node['B'] = BLUE[2]
    elif i + 5 in g_node:
        node['R'] = GREEN[0]
        node['G'] = GREEN[1]
        node['B'] = GREEN[2]
    elif i + 5 in r_node:
        node['R'] = RED[0]
        node['G'] = RED[1]
        node['B'] = RED[2]
    else:
        node['R'] = 0
        node['G'] = 0
        node['B'] = 0

    new_nodes.append(node)

new_nodes = pd.DataFrame(new_nodes)
new_nodes.to_csv('./reportdata/new_nodes.csv', index=False)

#--------------------------------------------------

edges = []
gen_list = [[1, 3], [2, 3], [4], [4]]

for i in range(node_num - 1):
    for j in gen_list[i]:
        edge = {}
        edge['u'] = node_id[i]
        edge['v'] = node_id[j]

        edges.append(edge)

edges = pd.DataFrame(edges)
edges.to_csv('./reportdata/2_edges.csv', index=False)

#---------------------------------------------------
new_edges = []

for i in range(node_num - 1):
    for j in gen_list[i]:
        edge = {}
        edge['u'] = node_id[i + 5]
        edge['v'] = node_id[j + 5]
        new_edges.append(edge)

new_edges = pd.DataFrame(new_edges)
new_edges.to_csv('./reportdata/new_edges.csv', index=False)

#---------------------------------------------------
mul_nodes = []
mul_nodes_loc = []
for item in base_location:
    mul_nodes_loc.append((item[0] + 3, item[1] + 3))

for k in range(3):
    for i in range(node_num):
        mul_node = {}
        mul_node['id'] = i + k * node_num
        mul_node['layer'] = k + 1
        mul_node['x'] = mul_nodes_loc[i][0]
        mul_node['y'] = mul_nodes_loc[i][1]
        mul_node['size'] = .3

        if k == 0:
            color = GREEN
        elif k == 1:
            color = BLUE
        elif k == 2:
            color = RED

        mul_node['R'] = color[0]
        mul_node['G'] = color[1]
        mul_node['B'] = color[2]

        mul_nodes.append(mul_node)

mul_nodes = pd.DataFrame(mul_nodes)
mul_nodes.to_csv('./reportdata/multilayernodes.csv', index=False)

#------------------------------------------------
mul_edges = []

for k in range(3):
    for i in range(node_num):
        for j in range(node_num):
            if i > j:
                edge = {}
                edge['u'] = i + k * node_num
                edge['v'] = j + k * node_num

                mul_edges.append(edge)

        if k >= 1:
            cedge = {}
            cedge['u'] = i + (k - 1) * node_num
            cedge['v'] = i + k * node_num
            cedge['style'] = 'dashed'
            mul_edges.append(cedge)

mul_edges = pd.DataFrame(mul_edges)
mul_edges.to_csv('./reportdata/multilayeredges.csv', index=False)

#----------------------------------------------------------------
comp_node_id = 'abcdefghijklm'
compress_edges = []
compress_nodes = []

comp_location = []
for item in node_location:
    comp_location.append((item[0] + 16, item[1] + 2))

for i in range(node_num):
    node = {}
    node['id'] = comp_node_id[i]
    node['size'] = .3
    node['x'] = comp_location[i][0]
    node['y'] = comp_location[i][1]

    compress_nodes.append(node)

    for j in range(node_num):
        if i > j:
            edge = {}
            edge['u'] = comp_node_id[i]
            edge['v'] = comp_node_id[j]

            compress_edges.append(edge)

compress_edges = pd.DataFrame(compress_edges)
compress_nodes = pd.DataFrame(compress_nodes)
compress_edges.to_csv('./reportdata/comedges.csv', index=False)
compress_nodes.to_csv('./reportdata/comnodes.csv', index=False)

#----------------------------------------------------------------
sparse_nodes = []
sparse_edges = []
sparse_location = []

tail = [0, 1, 1, 2, 2]
head = [1, 2, 4, 3, 4]

for item in comp_location:
    sparse_location.append((item[0] + 5, item[1]))

for i in range(node_num):
    node = {}
    node['id'] = 'c' + str(i)
    node['size'] = .3
    node['x'] = sparse_location[i][0]
    node['y'] = sparse_location[i][1]

    sparse_nodes.append(node)

    for j in range(node_num):
        if i > j:
            edge = {}
            edge['u'] = 'c' + str(i)
            edge['v'] = 'c' + str(j)

            sparse_edges.append(edge)

for item in sparse_edges:
    tu_item = (item['u'], item['v'])
    for edge_dash in zip(head, tail):
        edge_dash = ('c' + str(edge_dash[0]), 'c' + str(edge_dash[1]))
        if tu_item == edge_dash:
            item['style'] = 'dashed'

sparse_nodes = pd.DataFrame(sparse_nodes)
sparse_edges = pd.DataFrame(sparse_edges)
sparse_nodes.to_csv('./reportdata/sparsenodes.csv', index=False)
sparse_edges.to_csv('./reportdata/sparseedges.csv', index=False)
