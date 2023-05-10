import os
import glob
import pickle
import math
import numpy as np
from scipy.spatial import ConvexHull
from skspatial.objects import Plane, Line, LineSegment, Point, Points, Vector

epsilon = 1e-10


def gcd_3(a, b, c):
    return math.gcd(math.gcd(a, b), math.gcd(b, c))


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def lcm_3(a, b, c):
    return lcm(lcm(a, b), lcm(b, c))


def get_random_point_bfcc():
    sps = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.0, 0.5],
                    [0.5, 1.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [1.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.5, 1.0],
                    [0.5, 0.5, 0.5]])

    point = sps[np.random.choice(range(len(sps)))]
    return point


def point_in_array(point, array, epsilon=epsilon):
    return any([Point(p).distance_point(point) < epsilon for p in array])


def point_equal(point_1, point_2, epsilon=epsilon):
    return Point(point_1).distance_point(point_2) < epsilon


def where_pt_equal(point, value, epsilon=epsilon):
    return np.where(abs(point - value) < epsilon)[0]


def point_comp(point, value, epsilon=epsilon):
    return (abs(point - value) < epsilon).all()


def point_greater(point, value, epsilon=epsilon, inclusive=False):
    if inclusive:
        return (point >= value - epsilon).all()
    else:
        return (point > value - epsilon).all()


def point_smaller(point, value, epsilon=epsilon, inclusive=False):
    if inclusive:
        return (point <= value + epsilon).all()
    else:
        return (point < value + epsilon).all()


def arrays_equal(a1, a2, epsilon=epsilon):
    if a1.shape != a2.shape:
        return False
    else:
        return np.allclose(a1, a2, atol=epsilon)


def point_in_list(point, point_list, return_where=False):
    where = []
    for i in range(len(point_list)):
        if Point(point_list[i]).distance_point(point) < epsilon:
            where.append(i)
    if return_where:
        return where
    else:
        return True if len(where) != 0 else False


def array_in_list(array, list_arrays, epsilon=epsilon):
    return any([arrays_equal(array, a, epsilon) for a in list_arrays])


def line_in_line_list(line, line_list, return_where=False, epsilon=epsilon):
    where = []
    for i in range(len(line_list)):
        if line_list[i].is_close(line, abs_tol=epsilon):
            where.append(i)
    if return_where:
        return where
    else:
        return True if len(where) != 0 else False


def plane_in_plane_list(plane, plane_list, return_where=False, epsilon=epsilon):
    where = []
    for i in range(len(plane_list)):
        if plane_list[i].is_close(plane, abs_tol=epsilon):
            where.append(i)
    if return_where:
        return where
    else:
        return True if len(where) != 0 else False


def intersect_2_planes(plane_1, plane_2, epsilon=epsilon):
    if plane_1.is_close(plane_2, abs_tol=epsilon):
        intersection = plane_1
    else:
        try:
            intersection = plane_1.intersect_plane(plane_2)
        except ValueError:
            intersection = None
    return intersection


def intersect_2_lines(line_1, line_2, epsilon=epsilon):
    if line_1.is_close(line_2, abs_tol=epsilon):
        intersection = line_1
    else:
        try:
            intersection = line_1.intersect_line(line_2)
        except ValueError:
            intersection = None
    return intersection


def in_edge(p):
    if not point_greater(p, 0.0) or not point_smaller(p, 1.0):
        return False
    elif not any([point_comp(p1, 0.0) for p1 in p] + [point_comp(p1, 1.0) for p1 in p]):
        return False
    else:
        return True


def get_slope(line):
    if (line[1, 0] - line[0, 0]) == 0:
        m = np.inf
    else:
        m = (line[1, 1] - line[0, 1]) / (line[1, 0] - line[0, 0])
    return m


def index_in_edge(out_p):
    ix_vals = []
    for i in range(len(out_p)):
        if point_comp(out_p[i], 0.0):
            ix_vals.append((i, 0.0))
        if point_comp(out_p[i], 1.0):
            ix_vals.append((i, 1.0))
    return ix_vals


def pt_to_3d(pt, val, axis):
    if axis == "xy":
        return np.array([pt[0], pt[1], val])
    elif axis == "xz":
        return np.array([pt[0], val, pt[1]])
    elif axis == "yz":
        return np.array([val, pt[0], pt[1]])


def get_plane_from_poly(polygon, normal=None):
    if normal is None:
        normal = np.cross(polygon[0] - polygon[1], polygon[0] - polygon[2])
    return Plane(point=polygon[0], normal=normal)


def get_ordered_nodes(simplices):
    end = simplices[0][0]
    start = simplices[0][1]
    nodes = [end, start]
    lookup = simplices[1:]
    look = True
    while look is True:
        for i in lookup:
            if start in i:
                i.remove(start)
                if i[0] == end:
                    look = False
                    break
                else:
                    nodes.append(i[0])
                    start = i[0]
                    lookup.remove(i)
                    break
    return nodes


def to_2d(poly):
    """
    Returns the 2D projection of a 3d planar polygon.
    """
    # New reference system
    a = poly[1]-poly[0]
    a = a/np.linalg.norm(a)
    n = np.cross(a, poly[-1]-poly[0])
    n = n/np.linalg.norm(n)
    b = -np.cross(a, n)

    # Reference system change
    R_inv = np.linalg.inv(np.array([a, b, n])).T
    real = np.dot(R_inv, poly.T).T
    real[np.isclose(real, 0)] = 0

    return real[:, :2]


def get_edge_simplices(poly):
    poly = np.array(poly)
    hull = ConvexHull(to_2d(poly))
    simplices = hull.simplices

    return [[el for el in simplex] for simplex in simplices]


def arg_order_points_on_line(points, direction):
    first_point = points[0]
    distances = []
    for point in points:
        distances.append(np.dot(np.array(point) - np.array(first_point), direction))
    indexes = np.argsort(distances)

    return indexes


def get_edge_line_vertices(poly, edge_line):
    line_vertices = []
    for v in poly.vertices:
        if edge_line.distance_point(v) < epsilon:
            line_vertices.append(v)
    return np.array(line_vertices)


def line_in_polys(line, poly1, poly2):
    intersection_segment1 = []
    if len(poly1.vertices) == 1:
        if line.distance_point(poly1.vertices[0]) < epsilon:
            intersection_segment1.append(poly1.vertices[0])
    else:
        for edge in poly1.edge_lines:
            i_point = intersect_2_lines(edge, line)
            edge_points = get_edge_line_vertices(poly1, edge)
            if type(i_point) == Line:
                intersection_segment1 = [Point(e) for e in edge_points]
                break
            elif i_point is not None and not array_in_list(i_point, intersection_segment1) and \
                    LineSegment(*edge_points).contains_point(i_point, abs_tol=epsilon):
                intersection_segment1.append(i_point)

    intersection_segment2 = []
    if len(poly2.vertices) == 1:
        if line.distance_point(poly2.vertices[0]) < epsilon:
            intersection_segment2.append(poly2.vertices[0])

    else:
        for edge in poly2.edge_lines:
            i_point = intersect_2_lines(edge, line)
            edge_points = get_edge_line_vertices(poly2, edge)
            if type(i_point) == Line:
                intersection_segment2 = [Point(e) for e in edge_points]
                break
            elif i_point is not None and not array_in_list(i_point, intersection_segment2) and \
                    LineSegment(*edge_points).contains_point(i_point, abs_tol=epsilon):
                intersection_segment2.append(i_point)

    if len(intersection_segment1) == 0 or len(intersection_segment2) == 0:
        line_points = np.array([])
    elif len(intersection_segment1) == 1:
        line_points = np.array(intersection_segment1)
    elif len(intersection_segment2) == 1:
        line_points = np.array(intersection_segment2)
    elif intersection_segment1[0].distance_point(intersection_segment1[1]) < \
            intersection_segment2[0].distance_point(intersection_segment2[1]):
        line_points = np.array(intersection_segment1)
    else:
        line_points = np.array(intersection_segment2)

    return line_points


def write_pkl(var, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(var, f)


def read_plate_pkl(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_data(folder, data_dict):
    # Get the number for the next image by checking the number of files called plate_*.png where * is a number
    # in the image folder
    image_files = glob.glob(os.path.join(folder, "plate_*.png"))
    num = len(set([int(os.path.basename(f).split("_")[1].split(".")[0]) for f in image_files]))
    write_pkl(data_dict["plate_data"], os.path.join(folder, "plate_{}.pkl".format(num)))
    #save figure "plate_{}.png" in the image folder
    data_dict["plate"].savefig(os.path.join(folder, "plate_{}.png".format(num)),
                              format='png', transparent=True)
    data_dict["graph"].savefig(os.path.join(folder, "plate_{}_graph.png".format(num)),
                              format='png', transparent=True)
    for i, sub_poly in enumerate(data_dict["sub_polygons"]):
        sub_poly.savefig(os.path.join(folder, "plate_{}_sub_poly_{}.png".format(num, i)),
                         format='png', transparent=True)


def segment_intersect(ls_1, ls_2):
    x1, y1 = ls_1[0]
    x2, y2 = ls_1[1]
    x3, y3 = ls_2[0]
    x4, y4 = ls_2[1]
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:  # parallel
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    if ua < 0 or ua > 1:  # out of range
        return None
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    if ub < 0 or ub > 1:  # out of range
        return None
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return np.array([x, y])