import numpy as np
from skspatial.objects import Plane, Line, LineSegment, Point, Points, Vector
from scipy.spatial import ConvexHull
from fractions import Fraction
from itertools import combinations, product
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

uc = [Plane(point=[0, 0, 0], normal=[0, 1, 0]), Plane(point=[1, 1, 1], normal=[0, 1, 0]),
      Plane(point=[0, 0, 0], normal=[0, 0, 1]), Plane(point=[1, 1, 1], normal=[0, 0, 1]),
      Plane(point=[0, 0, 0], normal=[1, 0, 0]), Plane(point=[1, 1, 1], normal=[1, 0, 0])]
epsilon = 1e-10


def point_in_array(point, array, epsilon=1e-10):
    return any([Point(p).distance_point(point) < epsilon for p in array])


def point_equal(point_1, point_2, epsilon=1e-10):
    return Point(point_1).distance_point(point_2) < epsilon


def where_pt_equal(point, value, epsilon=1e-10):
    return np.where(abs(point - value) < epsilon)[0]


def point_comp(point, value, epsilon=1e-10):
    return (abs(point - value) < epsilon).all()


def point_greater(point, value, epsilon=1e-10, inclusive=False):
    if inclusive:
        return (point >= value - epsilon).all()
    else:
        return (point > value - epsilon).all()


def point_smaller(point, value, epsilon=1e-10, inclusive=False):
    if inclusive:
        return (point <= value + epsilon).all()
    else:
        return (point < value + epsilon).all()


def arrays_equal(a1, a2):
    if a1.shape != a2.shape:
        return False
    else:
        return np.allclose(a1, a2, atol=epsilon)


def array_in_list(array, list_arrays):
    return any([arrays_equal(array, a) for a in list_arrays])


def intersect_2_planes(plane_1, plane_2):
    if plane_1.is_close(plane_2):
        intersection = plane_1
    else:
        try:
            intersection = plane_1.intersect_plane(plane_2)
        except ValueError:
            intersection = None
    return intersection


def intersect_2_lines(line_1, line_2):
    if line_1.is_close(line_2):
        intersection = line_1
    else:
        try:
            intersection = line_1.intersect_line(line_2)
        except ValueError:
            intersection = None
    return intersection


def line_pts_in_uc(line):
    line_points = []

    for plane in uc:
        try:
            point_intersection = plane.intersect_line(line)
        except ValueError:
            point_intersection = None

        if point_intersection is not None:
            if not point_in_array(point_intersection, line_points) and \
                    point_greater(point_intersection, 0.0) and \
                    point_smaller(point_intersection, 1.0):
                line_points.append(point_intersection)
        if len(line_points) == 2:
            break
    return np.array(line_points)


def get_uc_plane_poly(uc_plane):
    plane_poly = np.zeros(shape=(4, 3))
    p_uc_plane = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    plane_i = np.argwhere(uc_plane.normal)[0][0]
    plane_poly[:, plane_i] = uc_plane.point[0]

    j = 0
    for i in [i for i in range(3) if i != plane_i]:
        plane_poly[:, i] = p_uc_plane[:, j]
        j += 1
    return plane_poly


def intersect_plane_uc(plane):
    uc_intersections = []
    poly_vertices = []
    vertices_uc = None

    for uc_plane in uc:
        intersection = intersect_2_planes(plane, uc_plane)
        if intersection is not None:
            if type(intersection) == Plane:
                vertices_uc = get_uc_plane_poly(uc_plane)

            elif type(intersection) == Line:
                line_vertices = line_pts_in_uc(intersection)
                for p in line_vertices:
                    if vertices_uc is not None  and point_greater(p, 0.0) and point_smaller(p, 1.0):
                        if not point_in_array(p, poly_vertices):
                            poly_vertices.append(p)
                        if not line_in_line_list(intersection, uc_intersections):
                            uc_intersections.append(intersection)

    if vertices_uc is not None:
        poly_vertices = vertices_uc
    return uc_intersections, np.array(poly_vertices)


def frac_to_float(frac):
    if type(frac) == float:
        return frac
    return float(frac.numerator/frac.denominator)


def sample_normal_for_inf_plate(max_int):
    p1 = np.random.choice(range(max_int))
    q1 = np.random.choice(range(max_int))

    p2 = np.random.choice(range(max_int))
    q2 = np.random.choice(range(max_int))

    p3 = np.random.choice(range(max_int))
    q3 = np.random.choice(range(max_int))

    m_xy = Fraction(p1, q1) if q1 != 0 else np.inf
    m_yz = Fraction(p3, q3) if q3 != 0 else np.inf

    if np.isinf(frac_to_float(m_xy)) and m_yz == 0:
        normal = np.array([p1 * p2, -q1 * p2, -q2 * p1])

    else:
        normal = np.array([p1 * p3, -q1 * p3, q1 * q3])

    if np.array_equiv(normal, [0.0, 0.0, 0.0]):
        normal = sample_normal_for_inf_plate(max_int)

    return normal


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

    p = sps[np.random.choice(range(len(sps)))]
    return p


def clean_lines(lines):
    line_lst = []
    for l in lines:
        unique_l = np.unique(l, axis=0)
        if unique_l.shape == (2, 2) and not array_in_list(unique_l, line_lst):
            line_lst.append(unique_l)

    return line_lst


def get_periodic_lines(vertices):
    lines = {'yz': [], 'xz': [], 'xy': []}

    for p_ix in ["xy", "xz", "yz"]:
        coord = list(lines.keys()).index(p_ix)
        lines_plane = []
        for val in [0., 1.]:
            plane_vertices = vertices[where_pt_equal(vertices[:, coord], val)]
            if len(plane_vertices) >= 2:
                if len(plane_vertices) == 4:
                    pass
                else:
                    line = plane_vertices[:, [i for i in range(3) if i != coord]]
                    lines_plane += get_parallel_lines(line)

        lines[p_ix] = clean_lines(lines_plane)

    return lines


def get_periodic_polygons(normal, vertices, lines):
    polygons = [vertices]
    for p_ix in ["xy", "xz", "yz"]:
        for val in [0., 1.]:
            for line in lines[p_ix]:
                p1 = pt_to_3d(line[0], val, p_ix)
                ls, vs = intersect_plane_uc(Plane(point=p1, normal=normal))
                if not array_in_list(vs, polygons):
                    polygons.append(vs)
    return polygons


def pt_to_3d(pt, val, axis):
    if axis == "xy":
        return np.array([pt[0], pt[1], val])
    elif axis == "xz":
        return np.array([pt[0], val, pt[1]])
    elif axis == "yz":
        return np.array([val, pt[0], pt[1]])


def get_parallel_lines(line):
    i = 0
    for p in line:
        if in_edge(p):
            i += 1
    if i < 2:
        return [line]

    m = get_slope(line)
    out_p = line[1]
    goal = go_through_uc(line[0], m)
    lines = [line]
    entry_p = go_through_uc(out_p, m)
    while not point_equal(out_p, goal, epsilon=1e-06) or point_equal(entry_p, goal, epsilon=1e-06):
        b = entry_p[1] - m * entry_p[0]
        out_p = get_out_p(entry_p, m, b)
        line = np.array([entry_p, out_p])
        lines.append(line)
        entry_p = go_through_uc(out_p, m)

    return lines


def in_edge(p):
    if not point_greater(p, 0.0) or not point_smaller(p, 1.0):
        return False
    elif not any([point_comp(p1, 0.0, epsilon=1e-10) for p1 in p] + [point_comp(p1, 1.0, epsilon=1e-10) for p1 in p]):
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
        if point_comp(out_p[i], 0.0, epsilon=1e-10):
            ix_vals.append((i, 0.0))
        if point_comp(out_p[i], 1.0, epsilon=1e-10):
            ix_vals.append((i, 1.0))
    return ix_vals


def go_through_uc(out_p, m):
    os = {0: 1.0, 1: 0.0}
    ix_val = index_in_edge(out_p)
    if m == 0:
        val = [val for (i, val) in ix_val if i == 0]
        if len(val) == 1:
            val = val[0]
            entry_p = np.array([os[val], out_p[1]])
        else:
            if np.copysign(1, m) > 0:
                entry_p = np.array([1., out_p[1]])
            else:
                entry_p = np.array([0., out_p[1]])

    elif m == np.inf:
        val = [val for (i, val) in ix_val if i == 1]
        if len(val) == 1:
            val = val[0]
            entry_p = np.array([out_p[0], os[val]])
        else:
            if np.copysign(1, m) > 0:
                entry_p = np.array([out_p[0], 1.])
            else:
                entry_p = np.array([out_p[0], 0.])

    elif set(ix_val) == {(0, 0.0), (1, 0.0)}:
        entry_p = np.array([1.0, 1.0])

    elif set(ix_val) == {(0, 1.0), (1, 1.0)}:
        entry_p = np.array([0.0, 0.0])

    elif set(ix_val) == {(0, 1.0), (1, 0.0)}:
        entry_p = np.array([0.0, 1.0])

    elif set(ix_val) == {(0, 0.0), (1, 1.0)}:
        entry_p = np.array([1.0, 0.0])

    else:
        ix = ix_val[0][0]
        val = ix_val[0][1]
        if ix == 0:
            entry_p = np.array([os[val], out_p[1]])
        else:
            entry_p = np.array([out_p[0], os[val]])
    return entry_p


def get_out_p(entry_p, m, b):
    os = {0: 1.0, 1: 0.0}
    try:
        x = os[entry_p[0]]
        y = m * x + b

        if y > 1:
            y = 1.0
            x = (y - b) / m
        if y < 0:
            y = 0.0
            x = (y - b) / m

    except KeyError:
        y = os[entry_p[1]]
        x = (y - b) / m

        if x > 1:
            x = 1.0
            y = m * x + b
        if x < 0:
            x = 0.0
            y = m * x + b

    out_p = np.array([x, y])

    return out_p


def clean_polys(polygons):
    poly_lst = []
    for p in polygons:
        unique_p = np.unique(p, axis=0)
        if unique_p.shape[0] > 2:
            poly_lst.append(unique_p)
    polys = []
    for i in range(len(poly_lst)):
        p = poly_lst[i]
        if not array_in_list(p, poly_lst[i+1:]):
            polys.append(p)
    return polys


def get_plane_from_poly(polygon, normal=None):
    if normal is None:
        normal = np.cross(polygon[0] - polygon[1], polygon[0] - polygon[2])
    return Plane(point=polygon[0], normal=normal)


def random_plane_for_semi_inf(normals):
    valid_planes = []
    for i in range(3):
        if len([n for n in normals if n[i] == 0]) > 0:
            valid_planes.append(i)

    plane = np.random.choice(valid_planes)

    return plane


def get_random_line_uc():
    line = Line.from_points(np.random.uniform(0, 1, size=2),
                            np.random.uniform(0, 1, size=2))

    if len(np.argwhere(abs(line.direction.unit()) < 0.05)) != 0:
        unit_dir = Vector([1., 1.])
        unit_dir[np.argwhere(abs(line.direction.unit()) < 0.05)[0]] = 0.0
        line = Line(point=line.to_point(0), direction=unit_dir)

    return line


def get_line_intersections_uc(line, lines_to_intersect):
    line_points = []

    for l in lines_to_intersect:
        try:
            point_intersection = l.intersect_line(line)
        except:
            continue

        if not point_in_array(point_intersection, line_points) and \
                point_smaller(point_intersection, 1.0, inclusive=True) and \
                point_greater(point_intersection, 0.0, inclusive=True):
            line_points.append(point_intersection)

    return line_points


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


def get_segment_intersections_uc(line, lines_to_intersect):
    line_points = []

    for l in lines_to_intersect:
        if len(l) != 2:
            continue
        point_intersection = segment_intersect(l, line)
        if point_intersection is None:
            continue
        if not point_in_array(point_intersection, line_points):
            line_points.append(point_intersection)

    return line_points


def get_semi_inf_line_pts(lines, d_min):
    uc_lines = [Line(point=[0, 0], direction=[0, 1]), Line(point=[0, 0], direction=[1, 0]),
                Line(point=[1, 1], direction=[0, 1]), Line(point=[1, 1], direction=[1, 0])]
    d = 0

    while d < d_min:
        periodic = False
        while True:
            semi_inf_line = get_random_line_uc()
            semi_inf_line_uc = np.array(get_line_intersections_uc(semi_inf_line, uc_lines))
            line_points = get_segment_intersections_uc(semi_inf_line_uc, lines)
            if len(line_points) > 0:
                break

        if len(line_points) >= 2:
            np.random.shuffle(line_points)
            semi_inf_line_pts = np.array([line_points[0], line_points[1]])
            periodic = True
        else:
            np.random.shuffle(semi_inf_line_uc)
            semi_inf_line_pts = np.array(
                [line_points[0], [p for p in semi_inf_line_uc if not np.array_equal(p, line_points[0])][0]])
            out_p = [p for p in semi_inf_line_uc if not np.array_equal(p, line_points[0])][0]
        d = Point(semi_inf_line_pts[0]).distance_point(semi_inf_line_pts[1])

    lines_for_periodicity = [semi_inf_line_pts]

    if periodic is False:
        m = get_slope(semi_inf_line_pts)
        entry_p = go_through_uc(out_p, m)

        stop = 0
        while stop == 0:
            line = Line(point=entry_p, direction=semi_inf_line.direction)
            line_uc = np.array(get_line_intersections_uc(line, uc_lines))
            int_points = get_segment_intersections_uc(line_uc, lines)
            if len(int_points) > 0:
                stop = 1
                if len(int_points) == 1:
                    lines_for_periodicity.append(np.array([entry_p, int_points[0]]))
                else:
                    i_min_dist = np.argmin([Point(entry_p).distance_point(p) for p in int_points])
                    if not np.array_equal(np.around(entry_p, 8), np.around(int_points[i_min_dist], 8)):
                        lines_for_periodicity.append(np.array([entry_p, int_points[i_min_dist]]))

            else:
                lines_for_periodicity.append(line_uc)
                out_p = [line_uc[i] for i in range(len(line_uc)) if
                         not np.array_equal(np.around(line_uc[i], 8), np.around(entry_p, 8))][0]
                entry_p = go_through_uc(out_p, m)

    return lines_for_periodicity


def get_semi_inf_polys(plane, semi_inf_lines_pts):
    ixs = [0, 1, 2]
    plane = ixs.pop(plane)
    polys = []
    for semi_inf_line_pts in semi_inf_lines_pts:
        poly_vs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        poly_vs[:2, ixs] = semi_inf_line_pts
        poly_vs[2:, ixs] = semi_inf_line_pts
        polys.append(poly_vs)

    normal = np.cross(poly_vs[0] - poly_vs[1], poly_vs[0] - poly_vs[2])
    return normal, polys


def plot_polygons(polygons, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    cmap = matplotlib.cm.get_cmap("tab20")
    colors = cmap(list(range(5, len(polygons) + 5)))

    for poly, c in zip(polygons, colors):
        try:
            ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], alpha=0.5, color=c)
        except:
            try:
                ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], triangles=[(0, 1, 2), (2, 3, 1)], alpha=0.5,
                                color=c)
            except:
                pass
        ax = plot_poly_edges(poly, ax=ax, color=c)

    return ax


def get_edge_simplices(poly):
    normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    new_poly = np.vstack((poly, 100 * normal))
    hull = ConvexHull(new_poly)
    simplices = []
    for s in hull.simplices:
        if len(poly) in s:
            simplices.append([i for i in s if i != len(poly)])
    return simplices


def get_poly_edges(poly):
    edge_sim = get_edge_simplices(poly)
    edges = [poly[e] for e in edge_sim]
    return edges


def plot_poly_edges(poly, ax=None, color=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_zticks([0, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    if color is None:
        color = "k"

    edges = get_poly_edges(poly)
    for e in edges:
        ax.plot(e[:, 0], e[:, 1], e[:, 2], c=color, linewidth=2)
    return ax


def line_in_polygons(line, poly1, poly2):
    intersection_segment1 = []
    for edge in get_poly_edges(poly1):
        e_line = Line.from_points(edge[0], edge[1])
        if e_line.direction.is_parallel(line.direction):
            continue
        else:
            try:
                i_point = e_line.intersect_line(line)
            except ValueError:
                i_point = None
            if i_point is not None:
                if not array_in_list(i_point, intersection_segment1) and \
                        LineSegment(edge[0], edge[1]).contains_point(i_point, abs_tol=epsilon):
                    intersection_segment1.append(i_point)
    intersection_segment2 = []
    for edge in get_poly_edges(poly2):
        e_line = Line.from_points(edge[0], edge[1])
        if e_line.direction.is_parallel(line.direction):
            continue
        else:
            try:
                i_point = e_line.intersect_line(line)
            except ValueError:
                i_point = None
            if i_point is not None:
                if not array_in_list(i_point, intersection_segment2) and \
                        LineSegment(edge[0], edge[1]).contains_point(i_point, abs_tol=epsilon):
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


def line_in_line_list(line, line_list, return_where=False):
    where = []
    for i in range(len(line_list)):
        if line_list[i].is_close(line, abs_tol=epsilon):
            where.append(i)
    if return_where:
        return where
    else:
        return True if len(where) != 0 else False


def point_in_list(point, point_list, return_where=False):
    where = []
    for i in range(len(point_list)):
        if Point(point_list[i]).distance_point(point) < epsilon:
            where.append(i)
    if return_where:
        return where
    else:
        return True if len(where) != 0 else False


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
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    if color is None:
        color = "k"
    for l in intersection_pts:
        if len(l) == 1 or arrays_equal(l[0], l[1]):
            ax.scatter(l[0, 0], l[0, 1], l[0, 2], c=color, linewidth=2)
        else:
            ax.plot(l[:, 0], l[:, 1], l[:, 2], c=color, linewidth=2)
    return ax


def to_2d(poly):
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


def area_poly(poly):
    return ConvexHull(to_2d(poly)).volume


def random_plane_uc():
    normal = Vector(np.random.rand(3)).unit()
    pt = Vector(np.random.rand(3))
    where_small = np.argwhere(normal < 0.1)
    where_big = np.argwhere(normal > 0.9)
    if len(where_small) > 0:
        normal[where_small[0]] = 0.0
    if len(where_big) > 0:
        normal[where_big[0]] = 1.0
    return Plane(normal=normal, point=pt)


def sample_random_plate():
    while True:
        plane = random_plane_uc()
        edge_lines, polygon = intersect_plane_uc(plane)
        area = area_poly(polygon)
        if area > 0.25:
            break
    return plane, edge_lines, polygon


def arg_order_points_on_line(points, direction):
    first_point = points[0]
    distances = []
    for point in points:
        distances.append(np.dot(np.array(point) - np.array(first_point), direction))
    indexes = np.argsort(distances)

    return indexes


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


def find_all_cycles(G, source=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes = [list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


def get_polygon(cyc, nodes, line_pt_ids, return_lines=False):
    node_list = [nodes[i] for i in cyc]
    simplices = [[cyc[i] for i in sim] for sim in get_edge_simplices(node_list)]
    sim_lines = [i for s in simplices for i in range(len(line_pt_ids)) if check_simplex_in_line(s, line_pt_ids[i])]
    all_in_lines = len(simplices) == len(sim_lines)
    if all_in_lines:
        if return_lines:
            return get_ordered_nodes(simplices), sim_lines
        else:
            return get_ordered_nodes(simplices)
    else:
        if return_lines:
            return None, None
        else:
            return None


def check_simplex_in_line(simplex, line):
    for i in simplex:
        if i not in line:
            return False
    return True


def get_sub_polygons_from_cycles(cycles, nodes, line_pt_ids):
    sub_polygons = []
    for cyc in cycles:
        poly, lines_in_poly = get_polygon(cyc, nodes, line_pt_ids, return_lines=True)
        if lines_in_poly is not None:
            sub_polygons.append([poly, lines_in_poly])
    return sub_polygons


def get_inside_sub_polygons(sub_polygons, lines_inside, min_area=None, nodes=None):
    inside_sub_polygons = []
    for (sub_poly, lines_in_sub_poly) in sub_polygons:
        if sum([1 for i in lines_in_sub_poly if not lines_inside[i]]) <= 1:
            if min_area is None:
                inside_sub_polygons.append([sub_poly, lines_in_sub_poly])
            else:
                if area_poly(np.array([nodes[i] for i in sub_poly])) > min_area:
                    inside_sub_polygons.append([sub_poly, lines_in_sub_poly])
    return inside_sub_polygons


def get_sub_polygon_cycles(polygon_intersections):
    nodes = polygon_intersections.all_points.to_dict()
    adj_matrix = polygon_intersections.get_adj_matrix()
    graph = nx.Graph(adj_matrix)
    cycles = find_all_cycles(graph)
    if len(cycles) > 1:
        cycles = [cyc for cyc in cycles if len(cyc) != len(graph.nodes)]
    # Get subpolygons
    sub_polys = get_sub_polygons_from_cycles(cycles, nodes, polygon_intersections.line_pt_ids())
    return sub_polys
