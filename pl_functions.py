from itertools import product
from fractions import Fraction
import networkx as nx
from copy import copy, deepcopy
from helpers import *

uc = [Plane(point=[0, 0, 0], normal=[0, 1, 0]), Plane(point=[1, 1, 1], normal=[0, 1, 0]),
      Plane(point=[0, 0, 0], normal=[0, 0, 1]), Plane(point=[1, 1, 1], normal=[0, 0, 1]),
      Plane(point=[0, 0, 0], normal=[1, 0, 0]), Plane(point=[1, 1, 1], normal=[1, 0, 0])]


def get_uc_plane_poly(uc_plane):
    """
    Returns the vertices of the polygon in the unit cell if the plane
    that defines it is one of the  unit cell planes.
    """

    plane_poly = np.zeros(shape=(4, 3))
    p_uc_plane = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    plane_i = np.argwhere(uc_plane.normal)[0][0]
    plane_poly[:, plane_i] = uc_plane.point[0]

    j = 0
    for i in [i for i in range(3) if i != plane_i]:
        plane_poly[:, i] = p_uc_plane[:, j]
        j += 1
    return plane_poly


def line_pts_in_uc(line):
    """
    Returns the points of intersection of a line with the unit cell planes.
    """
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


def intersect_plane_uc(plane):
    """
    Gets the intersections of an arbitrary plane with the unit cell planes.
    It returns the vertices of the polygon that results from this intersection.
    """
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
                    if point_greater(p, 0.0, inclusive=True) and point_smaller(p, 1.0, inclusive=True):
                        if vertices_uc is None and not point_in_array(p, poly_vertices):
                            poly_vertices.append(p)
                        if not line_in_line_list(intersection, uc_intersections) and len(line_vertices) == 2:
                            uc_intersections.append(intersection)

    if vertices_uc is not None:
        poly_vertices = vertices_uc
    return uc_intersections, np.array(poly_vertices)


def get_valid_vertices_from_neighbors(vertices, normal):
    """
    When the polygon is not fully in the unit cell, it will be defined by less than 3 vertices.
    This function gets the nearest neighbors of the vertices that are in the unit cell and
    returns the vertices of the polygon that results from the intersection of this new
    plane with the unit cell.
    """
    assert 3 > len(vertices) > 0, "vertices must be an array of 1 or 2 vertices"
    p0 = vertices[0]
    for idx in product([0, 1, -1], [0, 1, -1], [0, 1, -1]):
        new_p = p0 + idx

        if point_smaller(new_p, 1.0, inclusive=True) and point_greater(new_p, 0.0, inclusive=True):
            lines_uc, vertices = intersect_plane_uc(Plane(point=new_p, normal=normal))
            if len(vertices) >= 3:
                break
    return lines_uc, vertices


def get_vertices_and_lines_in_uc(poly):
    """
    Returns the vertices and lines of the polygon that are in the unit cell.
    poly: Polygon object
    """
    vertices = []
    edge_lines = []
    for v in poly.vertices:
        if point_greater(v, 0.0, inclusive=True) and point_smaller(v, 1.0, inclusive=True):
            vertices.append(v)
    vertices = np.array(vertices)

    if len(vertices) == 1:
        edge_lines = []
    elif len(vertices) == 2:
        for l in poly.edge_lines:
            if l.distance_point(vertices[0]) < epsilon and \
                    l.distance_point(vertices[1]) < epsilon:
                edge_lines = [l]
    elif len(vertices) > 2:
        edge_lines = poly.edge_lines
    return edge_lines, vertices


def clean_lines(lines):
    """
    Removes duplicate lines and lines that are not in the unit cell.
    """
    line_lst = []
    for l in lines:
        unique_l = np.unique(l, axis=0)
        if unique_l.shape == (2, 2) and not array_in_list(unique_l, line_lst):
            line_lst.append(unique_l)

    return line_lst


def go_through_uc(out_p, m):
    """
    Returns the point in the unit cell that is the entry point
    of a line that goes through the unit cell.
    """
    os = {0.0: 1.0, 1.0: 0.0}
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
    """
    Returns the exit point of a line of slope m that goes
    through the unit cell.
    """
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


def get_parallel_lines(line):
    """
    Returns the parallel lines needed so that a line
    is periodic in the unit cell plane.
    """
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


def get_periodic_lines(vertices):
    """
    Returns the periodic lines needed in the three unit cell planes
    so that a plane is periodic in the unit cell.
    """
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
    """
    Returns the periodic polygons needed so that a plane
    is periodic in the unit cell.
    """
    polygons = [vertices]
    polygon_edges = []
    for p_ix in ["xy", "xz", "yz"]:
        for val in [0., 1.]:
            for line in lines[p_ix]:
                p1 = pt_to_3d(line[0], val, p_ix)
                ls, vs = intersect_plane_uc(Plane(point=p1, normal=normal))
                if not array_in_list(vs, polygons):
                    polygons.append(vs)
                    polygon_edges.append(ls)
    return polygons, polygon_edges


def clean_polys(polygons):
    """
    Returns only the unique polygons without repeated vertices and eliminates lines.
    """
    poly_lst = []
    ixs = []
    for i in range(len(polygons)):
        unique_p = np.unique(polygons[i], axis=0)
        if unique_p.shape[0] > 2:
            poly_lst.append(unique_p)
            ixs.append(i)

    polys = []
    final_ixs = []
    for i in range(len(poly_lst)):
        p = poly_lst[i]
        if not array_in_list(p, poly_lst[i + 1:]):
            polys.append(p)
            final_ixs.append(ixs[i])
    return polys, final_ixs


def simplify_rational_normal(normal, limit_denominator=20, float_nums=False):
    """
    Returns the normal with the rational numbers simplified.
    """
    if len(np.argwhere(normal == 0)) == 2:
        normal[np.argwhere(normal != 0)[0][0]] = 1

    n1, n2, n3 = normal
    if not isinstance(n1, Fraction) and not isinstance(n2, Fraction) and not isinstance(n3, Fraction):
        n1 = Fraction(n1).limit_denominator(limit_denominator)
        n2 = Fraction(n2).limit_denominator(limit_denominator)
        n3 = Fraction(n3).limit_denominator(limit_denominator)
    gcd = gcd_3(n1.numerator, n2.numerator, n3.numerator)
    n1 /= gcd
    n2 /= gcd
    n3 /= gcd
    if float_nums:
        return np.array([float(n1), float(n2), float(n3)])
    else:
        return np.array([n1, n2, n3])


def plate_num_from_normal(normal, max_plates=20):
    """
    Returns the number of plates needed to fulfill the periodicity conditions of a
    plane with a given normal in the unit cell.
    max_plates: is the maximum number of plates to consider for the truncation of
    the normal into rational numbers.
    """
    limit_denominator = max_plates * 2
    n1, n2, n3 = simplify_rational_normal(normal, limit_denominator=limit_denominator)
    q = lcm_3(n1.denominator, n2.denominator, n3.denominator)
    p = int(abs(n1.numerator * q / n1.denominator) + abs(n2.numerator * q / n2.denominator)
            + abs(n3.numerator * q / n3.denominator))
    return p


def get_random_line_uc():
    """
    Returns a random line that goes through the unit cell.
    """
    line = Line.from_points(np.random.uniform(0, 1, size=2),
                            np.random.uniform(0, 1, size=2))

    if len(np.argwhere(abs(line.direction.unit()) < 0.05)) != 0:
        unit_dir = Vector([1., 1.])
        unit_dir[np.argwhere(abs(line.direction.unit()) < 0.05)[0]] = 0.0
        line = Line(point=line.to_point(0), direction=unit_dir)

    return line


def get_line_intersections_uc(line, lines_to_intersect):
    """
    Returns the points of intersection of a line within the unit cell.
    """
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


def get_segment_intersections_uc(line, lines_to_intersect):
    """
    Returns the points of intersection of a line with a line segment within the unit cell.
    """
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
    """
    Returns the points of the semi-infinite lines to create a semi-infinite plate that goes along a unit cell plane.
    """
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
                [line_points[0], [p for p in semi_inf_line_uc if not arrays_equal(p, line_points[0])][0]])
            out_p = [p for p in semi_inf_line_uc if not arrays_equal(p, line_points[0])][0]
        d = Point(semi_inf_line_pts[0]).distance_point(semi_inf_line_pts[1])

    line_pts = [semi_inf_line_pts]

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
                    line_pts.append(np.array([entry_p, int_points[0]]))
                else:
                    i_min_dist = np.argmin([Point(entry_p).distance_point(p) for p in int_points])
                    if not arrays_equal(entry_p, int_points[i_min_dist], epsilon=1e-8):
                        line_pts.append(np.array([entry_p, int_points[i_min_dist]]))

            else:
                line_pts.append(line_uc)
                out_p = [line_uc[i] for i in range(len(line_uc)) if
                         not arrays_equal(line_uc[i], entry_p, epsilon=1e-8)][0]
                entry_p = go_through_uc(out_p, m)

    clean_line_pts = []
    for l_pts in line_pts:
        if not point_equal(l_pts[0], l_pts[1]) and not array_in_list(l_pts, clean_line_pts):
            clean_line_pts.append(l_pts)

    return clean_line_pts, semi_inf_line.direction


def get_semi_inf_polys(plane, semi_inf_lines_pts, line_dir):
    """
    Returns the polygons of the semi-infinite plate along the uc.
    """
    assert len(semi_inf_lines_pts) > 0, "No semi-infinite lines to create a semi-infinite plate."
    ixs = [0, 1, 2]
    plane = ixs.pop(plane)
    free_edge_dir, uc_edge_dir = np.zeros(3), np.zeros(3)
    free_edge_dir[ixs] = line_dir
    uc_edge_dir[plane] = 1.0
    polys = []
    poly_edge_lines = []
    for semi_inf_line_pts in semi_inf_lines_pts:
        poly_vs = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        poly_vs[:2, ixs] = semi_inf_line_pts
        poly_vs[2:, ixs] = semi_inf_line_pts

        polys.append(poly_vs)
        poly_edge_lines.append([Line(point=poly_vs[:2][0], direction=free_edge_dir),
                                Line(point=poly_vs[2:][1], direction=free_edge_dir),
                                Line(point=poly_vs[2:][1], direction=uc_edge_dir),
                                Line(point=poly_vs[:2][0], direction=uc_edge_dir)])

    normal = np.cross(poly_vs[0] - poly_vs[1], poly_vs[0] - poly_vs[2])
    return normal, polys, poly_edge_lines


def random_plane_uc():
    """
    Samples a random plane in the unit cell where angles are not
    close to the horizontal and vertical.
    """
    normal = Vector(np.random.rand(3)).unit()
    pt = Vector(np.random.rand(3))
    where_small = np.argwhere(normal < 0.1)
    where_big = np.argwhere(normal > 0.9)
    if len(where_small) > 0:
        normal[where_small[0]] = 0.0
    if len(where_big) > 0:
        normal[where_big[0]] = 1.0
    return Plane(normal=normal, point=pt)


def area_poly(poly):
    """
    Returns the area of a polygon.
    """
    return ConvexHull(to_2d(poly)).volume


def sample_random_plate():
    """
    Samples a random plate in the unit cell and gets the polygon and edges defining it.
    """
    while True:
        plane = random_plane_uc()
        edge_lines, polygon = intersect_plane_uc(plane)
        area = area_poly(polygon)
        if area > 0.25:
            break
    return plane, edge_lines, polygon


def find_all_cycles(G, source=None):
    """Finds all cycles in a graph. Legacy function for the finite plate sampling"""
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


def edge_lines_simplex(simplex, polylines):
    """
    Checks if the nodes in a simplex are part of a polyline. Legacy function for finite plate sampling.
    """
    simplex_edge = None
    for polyline in polylines:
        node_ids = polyline.point_ids()
        if simplex[0] in node_ids and simplex[1] in node_ids:
            simplex_edge = polyline
            break

    return simplex_edge


def get_polygon(cyc, graph, polylines, return_lines=False):
    """
    Returns the polygon corresponding to the cycle cyc. If the cycle is for a non convex polygon it returns None.
    Legacy function for finite plate sampling.
    """
    node_list = [graph.nodes.data("point")[i] for i in cyc]
    simplices = [[cyc[i] for i in sim] for sim in get_edge_simplices(node_list)]
    cycle_edges = [edge_lines_simplex(s, polylines) for s in simplices]
    if all(cycle_edges):
        if return_lines:
            return get_ordered_nodes(simplices), [e.id for e in cycle_edges]
        else:
            return get_ordered_nodes(simplices)
    else:
        if return_lines:
            return None, None
        else:
            return None


def get_sub_polygons_from_cycles(cycles, graph, polylines):
    """
    Returns a list of the subpolygons in list of cycles.
    Legacy function for finite plate sampling.
    """
    sub_polygons = []
    for cyc in cycles:
        poly_nodes, poly_lines = get_polygon(cyc, graph, polylines, return_lines=True)
        if poly_lines is not None:
            sub_polygons.append([poly_nodes, poly_lines])
    return sub_polygons


def polygon_cycles(polylines, graph):
    """
    Gets all the cycles in a graph and then filters the ones that correspond to convex polygons.
    Legacy function for finite plate sampling.
    """
    cycles = find_all_cycles(graph)
    sub_polys = get_sub_polygons_from_cycles(cycles, graph, polylines)
    return sub_polys


def pts_to_match_bd_change(pts_to_match):
    """
    Change the points to match to the opposite boundary.
    :param pts_to_match:
    :return:
    """
    pts_bd_change = copy(pts_to_match)
    for i in range(3):
        for val in [0.0, 1.0]:
            where = where_pt_equal(pts_bd_change[:, i], val)
            if len(where) == len(pts_bd_change[:, i]):
                p = i
                value = [v for v in [0.0, 1.0] if v != val][0]

    pts_bd_change[:, p] = value
    return pts_bd_change


def order_cycle_nodes(G, cycle):
    """
    Orders the nodes in the cycle based on their position in the graph.

    Args:
        G (networkx.MultiGraph): the graph containing the cycle
        cycle (list): a list of nodes representing a cycle in G

    Returns:
        A list of nodes representing the ordered cycle.
    """
    ordered_cycle = []
    current_node = cycle[0]
    ordered_cycle.append(current_node)
    while len(ordered_cycle) < len(cycle):
        neighbors = list(G.neighbors(current_node))
        for neighbor in neighbors:
            if neighbor in cycle and neighbor not in ordered_cycle:
                current_node = neighbor
                ordered_cycle.append(current_node)
                break

    return ordered_cycle


def simplices_from_cycle(cycle):
    """
    Returns the simplices of a cycle.

    Args:
        cycle (list): a list of nodes representing a cycle in a graph

    Returns:
        A list of simplices representing the cycle.
    """
    simplices = []
    for i in range(len(cycle) - 1):
        simplices.append((cycle[i], cycle[i + 1]))
    simplices.append((cycle[-1], cycle[0]))
    return simplices


def get_sub_polys(poly_lines):
    """
    Returns the sub-polygons made up by intersection and edge lines in a polygon graph.
    :param poly_lines:
    :return:
    """
    poly_graph = poly_lines.get_graph()
    min_cycs = nx.minimum_cycle_basis(poly_graph)
    sub_polys = [order_cycle_nodes(poly_graph, cycle) for cycle in min_cycs]
    return sub_polys


def get_line_neighbor(line, idx):
    """
    Returns the neighbor of a line by a certain index.
    :param line:
    :param idx:
    :return:
    """
    return Line(point=line.point + idx, direction=line.direction)


def get_line_nn(line):
    """
    Returns the 27 nearest neighbors of a line.
    :param line:
    :return:
    """
    line_nn = [line]
    for idx in product([0, 1, -1], [0, 1, -1], [0, 1, -1]):
        if idx != (0, 0, 0):
            neighbor = get_line_neighbor(line, idx)
            if not line_in_line_list(neighbor, line_nn):
                line_nn.append(neighbor)
    return line_nn


def get_line_periodicity(line: Line):
    """
    Returns the periodic copies of a line.
    :param line:
    :return:
    """
    os = {0.0: 1.0, 1.0: 0.0}
    periodic_lines = [line]

    for l in get_line_nn(line):
        if not line_in_line_list(l, periodic_lines):
            periodic_lines.append(l)

    for plane in uc:
        if line.direction.is_parallel(plane.normal):
            continue
        if plane.project_line(line).is_close(line):
            new_line = deepcopy(line)
            new_line.point[[np.argwhere(plane.normal)[0]][0]] = os[plane.point[np.argwhere(plane.normal)[0]][0]]
            if not line_in_line_list(new_line, periodic_lines):
                periodic_lines.append(new_line)

    return periodic_lines



