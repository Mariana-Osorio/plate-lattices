import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from netgraph import Graph
from sample_pl import *


def plot_polygons_uc(vert_list, ax=None, colors=None):
    """
    Plots the polygons in the unit cell.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)
        ax.set_box_aspect(aspect=(1, 1, 1))

    cmap = matplotlib.cm.get_cmap("tab20")
    colors = cmap(list(range(5, len(vert_list) + 5))) if colors is None else colors

    for vert, c in zip(vert_list, colors):
        if len(vert) > 2:
            p3c = Poly3DCollection([vert[get_ordered_nodes(get_edge_simplices(vert))]],
                                   alpha=0.5, facecolors=[c], edgecolors=[c],
                                   linewidth=2)
            ax.add_collection3d(p3c)
        elif len(vert) == 2:
            ax.plot(vert[:, 0], vert[:, 1], vert[:, 2], color=c, linewidth=2)

        elif len(vert) == 1:
            ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], color=c, linewidth=2)

    return ax


def plot_intersections(intersection_pts, ax=None, color=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_zlabel('z', fontsize=16)
        ax.set_box_aspect(aspect=(1, 1, 1))

    if color is None:
        color = "k"
    for l in intersection_pts:
        if len(l) == 1 or arrays_equal(l[0], l[1]):
            ax.scatter(l[0, 0], l[0, 1], l[0, 2], c=color, linewidth=2, zorder=100)
        else:
            ax.plot(l[:, 0], l[:, 1], l[:, 2], c=color, linewidth=2, zorder=100)
    return ax


def draw_graph(G):
    edge_labels = {}
    for e in G.edges:
        if e[:2] in edge_labels:
            edge_labels[e[:2]].append(G.edges[e]["id"])
        else:
            edge_labels[e[:2]] = [G.edges[e]["id"]]

    G = nx.Graph(G)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    Graph(G, node_layout="spring", node_size=5, node_labels=True, node_label_fontdict={"size": 18},
          edge_labels=edge_labels, edge_label_fontdict={"size": 18}, ax=ax)
    return ax


def plot_polygon_intersections(pl_uc: PlateList,
                               intersection_list: IntersectionList,
                               poly_ix, id_dict=None, ax=None):
    cmap = matplotlib.cm.get_cmap("tab20")
    colors = cmap(list(range(5, len(pl_uc.polygon_list()) + 5)))
    poly = deepcopy(pl_uc.polygon_list()[poly_ix])
    color = colors[poly_ix]
    if id_dict is None:
        pid = poly.id
    else:
        pid = id_dict[poly_ix] if poly_ix in id_dict else poly.id
    polylines = get_poly_lines(poly, intersection_list, pid)
    int_lines = get_intersection_line_pts_from_poly_lines(polylines)
    normal = Vector(poly.normal).unit()
    elev = np.arcsin(normal[2]) * 180 / np.pi
    azim = np.arctan2(normal[1], normal[0]) * 180 / np.pi

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.figure.set_figwidth(5)
        ax.figure.set_figheight(5)

    ax = plot_polygons_uc([poly.vertices], colors=[color], ax=ax)
    ax = plot_intersections(int_lines, ax=ax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.axis('off')
    return ax

def plot_lines(lines, m=None, ax=None, plane=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    if m:
        ax.set_title(f"{len(lines)} line(s), m = {m}", fontsize=20)
    else:
        ax.set_title("%i line(s)" % (len(lines)), fontsize=20)
    if plane:
        ax.set_xlabel(plane[0], fontsize=20)
        ax.set_ylabel(plane[1], fontsize=20)
    else:
        ax.set_xlabel('x', fontsize=20)
        ax.set_ylabel('y', fontsize=20)

    for line in lines:
        ax.plot(line[:, 0], line[:, 1], linewidth = 4)

    return ax


def get_nice_ax(ax):
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.set_zlabel('z', fontsize=16)
    ax.set_box_aspect(aspect=(1, 1, 1))
    return ax
