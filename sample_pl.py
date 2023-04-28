import warnings
from PlateLattices import *

# Infinite plates
def sample_normal_for_inf_plate(max_plates):
    """
    Sample a random normal with rational components for an infinite plate with the max_plates condition.
    :param max_plates: maximum polygons for the infinite plate
    """
    max_int = max_plates * 2
    while True:
        n1 = np.random.choice((-1, 1)) * Fraction(np.random.randint(0, max_int), np.random.randint(1, max_int))
        n2 = np.random.choice((-1, 1)) * Fraction(np.random.randint(0, max_int), np.random.randint(1, max_int))
        n3 = np.random.choice((-1, 1)) * Fraction(np.random.randint(0, max_int), np.random.randint(1, max_int))
        if n1 == 0 and n2 == 0 and n3 == 0:
            continue
        if plate_num_from_normal([n1, n2, n3], max_plates=max_plates) <= max_plates:
            break

    return np.array([float(n1), float(n2), float(n3)])


def inf_plate(normal, p):
    """
    Returns the polygons and polygon edges of an infinite plate with normal and point p.
    :param normal: must have rational components or else the parallel_line
    function will be stuck in an infinite loop.
    :param p: point in the unit cell
    :return: polygons, polygon_edges
    """
    normal = np.array(normal)
    p = np.array(p)
    assert point_greater(p, 0.0, inclusive=True) and point_smaller(p, 1.0, inclusive=True), \
        "point must be in unit cell"
    lines_uc, vertices = intersect_plane_uc(Plane(point=p, normal=normal))
    if len(vertices) in [1, 2]:
        lines_uc, vertices = get_valid_vertices_from_neighbors(vertices, normal)
    lines = get_periodic_lines(vertices)
    polygons, polygon_edges = get_periodic_polygons(normal, vertices, lines)
    polygon_edges = [lines_uc] + polygon_edges
    polygons, ixs = clean_polys(polygons)
    polygon_edges = [polygon_edges[i] for i in ixs]
    return polygons, polygon_edges


def sample_inf_plate(max_plates=5, position="random"):
    """
    Sample a random infinite plate with the max_plates condition. Returns normal,
    polygons and edge lines.
    :param max_plates: maximum polygons for the infinite plate
    :param position: point in the unit cell to place the first polygon,
    valid options are "random" or "bfcc":
        "random" chooses a uniform random point in the unit cell
        "bfcc" samples from the points in the fcc and bcc unit cell
    :return: normal, polygons, polygon_edges
    """
    while True:
        normal = sample_normal_for_inf_plate(max_plates)
        p = np.around(np.random.rand(3), 1) if position != "bfcc" else get_random_point_bfcc()
        polygons, polygon_edges = inf_plate(normal, p)
        if max_plates >= len(polygons) > 0:
            break
    return normal, polygons, polygon_edges


def inf_plate_from_normal_and_point(normal, p):
    """
    Returns an infinite plate Plate Object from a normal and point p.
    :param normal: must have rational components or else the parallel_line function
    will be stuck in an infinite loop.
    :param p: point in the unit cell
    :return: plate
    """
    normal = np.array(normal)
    p = np.array(p)
    assert point_greater(p, 0.0, inclusive=True) and point_smaller(p, 1.0, inclusive=True), \
        "point must be in unit cell"
    polygons, polygon_edges = inf_plate(normal, p)
    plate = Plate(normal, PolygonList.from_lists(polygons, polygon_edges, normal), "inf")

    return plate

# Rational semi-infinite plates
def normal_for_rational_semi_inf_plate(inf_plate_normal, num_plates=2):
    """
    Returns a normal for a rational semi-infinite plate.
    :param inf_plate_normal: normal of the infinite plate the
    semi-infinite plate will hold on to.
    :param num_plates: number of independent sub-polygons that the
    plates will subdivide in to choose from afterwards.
    :return: new_normal
    """
    inf_plate_normal = simplify_rational_normal(inf_plate_normal, float_nums=True)
    new_normal = np.zeros(3)
    choice = np.random.choice(3)
    coords = [[0, 1, 2], [1, 2, 0], [0, 2, 1]]
    ixs = coords[choice]
    if np.array_equal(inf_plate_normal[ixs[:2]], np.zeros(2)):
        ixs = coords[list({0, 1, 2} - {choice})[np.random.choice(2)]]
    new_normal[ixs[:2]] = inf_plate_normal[ixs[:2]]
    plate_nums = []
    denominators = [den for den in list(range(-num_plates * 3, num_plates * 3 + 1)) if den % num_plates]
    for den in denominators:
        new_normal[ixs[2]] = inf_plate_normal[ixs[2]] + num_plates / den
        plate_nums.append(plate_num_from_normal(new_normal, max_plates=num_plates))
    i = np.argmin(plate_nums)
    new_normal[ixs[2]] = inf_plate_normal[ixs[2]] + num_plates / denominators[i]
    return new_normal


def plate_without_random_polygon(valid_sub_polys_plate):
    """
    When an infinite plate subdivides in more than one independent sub-polygons, this
    function can be used to choose randomly one of these sub-polygons.
    :param valid_sub_polys_plate: valid independent sub-polys of the infinite plate
    :return: polygons_list, polygon_edges_list
    """
    sub_polys = valid_sub_polys_plate[np.random.choice(len(valid_sub_polys_plate))]
    polygons_list, polygon_edges_list = [], []
    for valid_sub_poly in sub_polys:
        nodes = valid_sub_poly[0].get_graph().nodes
        vertices = []
        edges_list = []
        for n in valid_sub_poly[1]:
            vertices.append(nodes[n]["point"])

        for i in range(len(valid_sub_poly[1])):
            i1 = 0 if i + 1 == len(valid_sub_poly[1]) else i + 1
            edge = valid_sub_poly[0].get_edge([valid_sub_poly[1][i], valid_sub_poly[1][i1]])
            edge_line = valid_sub_poly[0][edge.line_id].line
            edges_list.append(edge_line)

        polygons_list.append(np.array(vertices))
        polygon_edges_list.append(edges_list)

    return polygons_list, polygon_edges_list


def semi_inf_plate_at_rational_normal(new_normal, new_point, plate_to_hold):
    """
    Returns a semi-infinite plate at a rational normal and point.
    :param new_normal: normal for the semi-infinite plate, must be rational
    :param new_point: point for the semi-infinite plate.
    :param plate_to_hold: other plate in the plate-lattice with which it will intersect
    :return: semi_inf_plate
    """
    sim_normal = simplify_rational_normal(plate_to_hold.normal, float_nums=True)
    assert (sim_normal == new_normal).sum() == 2, \
        f"Normal not valid for semi_inf plate. Inf: {sim_normal}, semi inf: {new_normal}"
    diff = sim_normal - new_normal
    assert abs(Fraction(diff[np.where(diff)[0][0]]).limit_denominator(20).numerator) >= 2, \
        f"Normal not valid for semi_inf plate. Inf: {sim_normal}, semi inf: {new_normal}"
    new_plate = inf_plate_from_normal_and_point(new_normal, new_point)
    temp_pl = PlateList([plate_to_hold, new_plate])
    temp_pl_nn, temp_pl_nn_dict = temp_pl.get_nn_uc_plates(return_dict=True)
    temp_intersection_list = IntersectionList.from_plate_list(temp_pl_nn)
    valid_sub_polys_plate = get_plate_valid_sub_polys(temp_pl, temp_intersection_list, id_dict=temp_pl_nn_dict)
    polygons_list, polygon_edges_list = plate_without_random_polygon(valid_sub_polys_plate[1])
    semi_inf_plate = Plate(new_normal, PolygonList.from_lists(polygons_list, polygon_edges_list, new_normal),
                           "semi_inf")
    return semi_inf_plate


def sample_semi_inf_rational_plate(plate_to_hold, num_plates=2, position="random"):
    """
    Returns a random rational semi-infinite plate holding on to plate_to_hold.
    :param plate_to_hold: other plate in the plate-lattice with which it will intersect
    :param num_plates: maximum number of sub-polygons in the semi-infinite plate
    :param position: how to sample the point where to put the plate,
    can be "random" or "bfcc"
    :return: semi_inf_plate
    """
    assert num_plates >= 2, "num_plates must be greater or equal than 2"
    new_normal = normal_for_rational_semi_inf_plate(plate_to_hold.normal, num_plates=num_plates)
    new_point = np.around(np.random.rand(3), 1) if position != "bfcc" else get_random_point_bfcc()
    semi_inf_plate = semi_inf_plate_at_rational_normal(new_normal, new_point, plate_to_hold)

    return semi_inf_plate


# Semi-infinite plate along the unit cell
def valid_uc_planes_for_semi_inf_plate_along_uc(pl_normals):
    """
    Returns the valid unit cell planes for a semi-infinite plate along the unit cell.
    :param pl_normals: normals of the plates in the plate-lattice
    :return: valid_uc_planes
    """
    valid_uc_planes = {}

    for i in range(len(pl_normals)):
        for j in range(3):
            if pl_normals[i][j] == 0:
                if j not in valid_uc_planes:
                    valid_uc_planes[j] = [i]
                else:
                    valid_uc_planes[j].append(i)

    return valid_uc_planes


def sample_semi_inf_plate_along_uc(pl_uc, d_min=0.3, max_plates=5):
    """
    Returns a random semi-infinite plate along the unit cell.
    :param pl_uc: plate-lattice
    :param d_min: minimum distance between plates
    :param max_plates: maximum number of sub-polygons in the semi-infinite plate
    :return: semi_inf_plate
    """
    valid_uc_planes = valid_uc_planes_for_semi_inf_plate_along_uc(pl_uc.normal_list())
    plane = list(valid_uc_planes.keys())[np.random.choice(len(valid_uc_planes))]
    valid_polys = valid_uc_planes[plane]
    lines_for_semi_inf = pl_uc.get_uc_plane_lines(poly_ixs=valid_polys)[['yz', 'xz', 'xy'][plane]]

    while True:
        semi_inf_lines_pts, semi_inf_line_dir = get_semi_inf_line_pts(lines_for_semi_inf, d_min)
        normal, semi_inf_polys, semi_inf_poly_edge_lines = get_semi_inf_polys(plane, semi_inf_lines_pts,
                                                                              semi_inf_line_dir)
        if len(semi_inf_polys) <= max_plates:
            break

    semi_inf_plate = Plate(normal, PolygonList.from_lists(semi_inf_polys, semi_inf_poly_edge_lines, normal),
                           "semi_inf")
    return semi_inf_plate


# Plate lattices
def base_pl(max_plates=5, position="random"):
    """
    Returns a random base plate-lattice with two infinite plates.
    :param max_plates: maximum number of sub-polygons in the semi-infinite plate
    :param position: where to put the plate, can be "random" or "bfcc"
    :return: pl
    """
    pl = PlateList([])
    for i in range(2):
        while True:
            normal, polygons, polygon_edges = sample_inf_plate(max_plates=max_plates, position=position)
            if not pl.has_normal(normal):
                break
        plate = Plate(normal, PolygonList.from_lists(polygons, polygon_edges, normal), "inf")
        pl.add_plate(plate)

    return pl


def inf_pl_from_lists(normal_list, p_list):
    """
    Returns a plate-lattice made up of n infinite plates from a list of normals and points.
    :param normal_list: list of size n of the n normals of the infinite plates
    :param p_list: list of size n of the n points of the infinite plates
    :return: pl
    """
    normal_list = [np.array(normal) for normal in normal_list]
    p_list = [np.array(p) for p in p_list]
    pl = PlateList([])
    for normal, p in zip(normal_list, p_list):
        polygons, polygon_edges = inf_plate(normal, p)
        if not pl.has_plane(get_plane_from_poly(polygons[0], normal=normal)):
            pl.add_plate(Plate(normal, PolygonList.from_lists(polygons, polygon_edges, normal), "inf"))

    assert len(np.unique(np.array(pl.normal_list()), axis=0)) >= 2, \
        "Plate lattice is not connected."
    return pl


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


def check_simplex_in_line(simplex, line):
    for i in simplex:
        if i not in line:
            return False
    return True


def edge_lines_simplex(simplex, polylines):
    simplex_edge = None
    for polyline in polylines:
        node_ids = polyline.point_ids()
        if simplex[0] in node_ids and simplex[1] in node_ids:
            simplex_edge = polyline
            break

    return simplex_edge


def get_polygon(cyc, graph, polylines, return_lines=False):
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
    sub_polygons = []
    for cyc in cycles:
        poly_nodes, poly_lines = get_polygon(cyc, graph, polylines, return_lines=True)
        if poly_lines is not None:
            sub_polygons.append([poly_nodes, poly_lines])
    return sub_polygons


def polygon_cycles(polylines, graph):
    cycles = find_all_cycles(graph)
    sub_polys = get_sub_polygons_from_cycles(cycles, graph, polylines)
    return sub_polys


def only_num_edges_outside(sub_poly, polylines, num_edges=1):
    nodes, lines = sub_poly
    all_edges = []
    for i in range(len(nodes)):
        i0 = nodes[i]
        i1 = nodes[i + 1] if (i + 1) < len(nodes) else nodes[0]
        for l_id in lines:
            if i0 in polylines[l_id].point_ids() and i1 in polylines[l_id].point_ids():
                line = polylines[l_id]
                edge_start, edge_end = sorted([line.point_ids().index(i0), line.point_ids().index(i1)])
                line_edges = line.edges[edge_start:edge_end]
        all_edges.append(line_edges)
    lines_inside = [all([e.is_intersection for e in edge]) for edge in all_edges]
    return len(lines_inside) - sum(lines_inside) <= num_edges


def finite_plate_polylines_and_graph(finite_plate, pl_uc):
    pl_uc_nn, uc_nn_dict = pl_uc.get_nn_uc_plates(return_dict=True)
    temp_pl = PlateList([plate for plate in pl_uc_nn] + [finite_plate])
    temp_intersection_list = IntersectionList.from_plate_list(temp_pl)

    polylines = get_poly_lines(temp_pl[-1][0], temp_intersection_list, temp_pl[-1][0].id)
    graph = polylines.get_graph()

    return polylines, graph


def index_points_from_graph(ixs, graph):
    return np.array([graph.nodes.data("point")[n_id] for n_id in ixs])


def sample_finite_polygon(pl_uc):
    """
    Sample a random normal and point to create a plane in the unit cell. Then sample a random polygon from the
    intersection of the plane with the unit cell. This polygon must have at most only one outside edge that is not an
    intersection with the plate lattice. It will return the finite polygon and if there is an outside edge, the points
    the next polygon should match in the opposite boundary.
    :param pl_uc: plate lattice unit cell
    :return: finite_polygon, pts_to_match
    """
    inside_sub_polys = []
    i = 0
    while len(inside_sub_polys) <= 1:
        if i > 4:
            return None, None
        plane, polygon_edges, polygons = sample_random_plate()
        new_plate = Plate(plane.normal, PolygonList.from_lists([polygons], [polygon_edges], plane.normal), "finite")
        polylines, graph = finite_plate_polylines_and_graph(new_plate, pl_uc)
        sub_polys = polygon_cycles(polylines, graph)
        inside_sub_polys = [sub_poly for sub_poly in sub_polys if
                            only_num_edges_outside(sub_poly, polylines, num_edges=1)]
        i += 1

    j = np.random.choice(range(len(inside_sub_polys)))
    finite_plate_normal = plane.normal
    finite_poly = index_points_from_graph(inside_sub_polys[j][0], graph)
    finite_poly_lines = polylines.polyline_subset_sub_polygon(inside_sub_polys[j])
    finite_polygon = Polygon(finite_poly, [l.line for l in finite_poly_lines], finite_plate_normal)
    l_outside_ix = np.argwhere([all([not edge.is_intersection for edge in l.edges]) for l in finite_poly_lines])

    if len(l_outside_ix) == 0:
        pts_to_match = None
    else:
        l_outside = finite_poly_lines[l_outside_ix[0][0]]
        pts_to_match = line_in_polys(l_outside.line, finite_polygon, finite_polygon)

    return finite_polygon, pts_to_match


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


def get_next_poly_cycles(normal, pts_to_match, pl_uc):
    """
    Get the next polygon cycles for the next polygon in the finite plate.
    :param normal: normal of the finite plate.
    :param pts_to_match: points to match from previous polygon.
    :param pl_uc: plate lattice unit cell.
    :return: valid_sub_poly_cycles, polylines, graph
    """
    pts_to_match = pts_to_match_bd_change(pts_to_match)
    plane_to_match = Plane(normal=normal, point=pts_to_match[0])
    edge_lines, polygon = intersect_plane_uc(plane_to_match)
    new_plate = Plate(normal, PolygonList.from_lists([polygon], [edge_lines], normal), "finite")
    polylines, graph = finite_plate_polylines_and_graph(new_plate, pl_uc)
    wanted_nodes = set([n_id for n_id, point in graph.nodes.data("point") if
                        arrays_equal(point, pts_to_match[0]) or
                        arrays_equal(point, pts_to_match[1])])
    sub_polys = [[sp, sp_lines] for (sp, sp_lines) in polygon_cycles(polylines, graph) if
                 set(sp).intersection(set(wanted_nodes)) == set(wanted_nodes)]
    valid_sub_poly_cycles = [sub_poly for sub_poly in sub_polys if
                             only_num_edges_outside(sub_poly, polylines, num_edges=2)]

    return valid_sub_poly_cycles, polylines, graph


def sample_next_poly(normal, pts_to_match, pl_uc):
    """
    Sample the next polygon in the finite plate. It first gets the next polygon cycles and then chooses one of
    the subpolygons randomly. It afterwards returns the polygon object of the sub-polygon of the finite plane
     as well as the next points to match if it has an outside edge.
    :param normal: normal of the finite plate.
    :param pts_to_match: points to match from previous polygon.
    :param pl_uc: plate lattice unit cell.
    :return: next_polygon, pts_to_match
    """
    valid_sub_poly_cycles, polylines, graph = get_next_poly_cycles(normal, pts_to_match, pl_uc)
    if len(valid_sub_poly_cycles) == 0:
        return None, None
    else:
        j = np.random.choice(len(valid_sub_poly_cycles))
        next_plate_normal = normal
        next_poly_vertices = index_points_from_graph(valid_sub_poly_cycles[j][0], graph)
        next_poly_lines = polylines.polyline_subset_sub_polygon(valid_sub_poly_cycles[j])
        next_poly_graph = next_poly_lines.get_graph()
        next_polygon = Polygon(next_poly_vertices, [l.line for l in next_poly_lines], next_plate_normal)

        #Find if there is an outside edge
        ok_edge = pts_to_match_bd_change(pts_to_match)
        ok_nodes = set([n_id for n_id, point in next_poly_graph.nodes.data("point") if
                        arrays_equal(point, ok_edge[0]) or
                        arrays_equal(point, ok_edge[1])])

        l_outside_ix = np.argwhere([all([not edge.is_intersection for edge in line.edges]) and
                                    not set(ok_nodes).intersection(set(line.point_ids())) == set(ok_nodes)
                                    for line in next_poly_lines])

        if len(l_outside_ix) == 0:
            pts_to_match = None
        else:
            l_outside = next_poly_lines[l_outside_ix[0][0]]
            pts_to_match = line_in_polys(l_outside.line, next_polygon, next_polygon)

    return next_polygon, pts_to_match


def sample_finite_plate(pl_uc, max_plates=5, tries=5):
    """
    Returns a random finite polygon that will hold on to the plate lattice.
    :param pl_uc: plate lattice unit cell.
    :param max_plates: maximum number of subpolygons in the finite plate.
    :param tries: maximum number of tries to get a finite plate.
    :return: finite_plate
    """
    for i in range(tries):
        finite_sub_polygons = []
        finite_polygon, pts_to_match = sample_finite_polygon(pl_uc)
        if finite_polygon is None:
            continue
        else:
            finite_sub_polygons.append(finite_polygon)
            if pts_to_match is not None:
                next_iter = False
                for j in range(max_plates-1):
                    next_polygon, pts_to_match = sample_next_poly(finite_polygon.normal, pts_to_match, pl_uc)
                    if next_polygon is None:
                        next_iter = True
                        break
                    else:
                        finite_sub_polygons.append(next_polygon)
                        finite_polygon = copy(next_polygon)
                        if pts_to_match is None:
                            break
                if next_iter:
                    continue
            break
    if i == tries-1:
        return None
    else:
        return Plate(finite_sub_polygons[0].normal, PolygonList(finite_sub_polygons), typ="finite")


def pl_from_lists(normal_list, polygons_list, polygon_edges_list, plate_type_list):
    """
    Returns a plate-lattice from a list of normals, polygons, polygon edges and plate types.
    :param normal_list: list of size n of the n normals of the plates
    :param polygons_list: list of size n of the n polygons of the plates
    :param polygon_edges_list: list of size n of the n polygon edges of the plates
    :param plate_type_list: list of size n of the n plate types of the plates
    :return:
    """
    assert len([p_type for p_type in plate_type_list if p_type == "inf"]) >= 2, \
        "Plate lattice must have at least 2 infinite plates."
    assert len(np.unique([normal_list[i] for i in range(len(normal_list))
                          if plate_type_list[i] == "inf"], axis=0)) >= 2, \
        "Plate lattice is not connected."
    pl = PlateList([])
    for normal, polygons, polygon_edges, plate_type in zip(normal_list, polygons_list, polygon_edges_list,
                                                           plate_type_list):
        if not pl.has_plane(get_plane_from_poly(polygons[0], normal=normal)):
            pl.add_plate(Plate(normal, PolygonList.from_lists(polygons, polygon_edges, normal), plate_type))

    assert len(np.unique(np.array(pl.normal_list()), axis=0)) >= 2, \
        "Plate lattice is not connected."

    return pl


def add_inf_plate(pl, max_plates=5, position="random"):
    """
    Adds a random infinite plate to the plate-lattice.
    :param pl: plate-lattice
    :param max_plates: maximum number of sub-polygons in the infinite plate
    :param position: where to put the plate, can be "random" or "bfcc"
    :return: pl
    """
    while True:
        normal, polygons, polygon_edges = sample_inf_plate(max_plates=max_plates, position=position)
        if not pl.has_plane(Plane(normal=normal, point=polygons[0][0])):
            break
    plate = Plate(normal, PolygonList.from_lists(polygons, polygon_edges, normal), "inf")
    pl.add_plate(plate)
    return pl


def add_semi_inf_plate(pl_uc, typ="rational", position="random", num_plates=2, d_min=0.3, max_plates=5):
    """
    Adds a random semi-infinite plate to the plate-lattice.
    :param pl_uc: plate-lattice
    :param typ: type of semi-infinite plate, can be "rational" or "along_uc"
    :param position: where to put the plate, can be "random" or "bfcc"
    :param num_plates: number of independent sub-polygons in the rational
    semi-infinite plate
    :param d_min: minimum distance between the semi-infinite plate and the other plates
    :param max_plates: maximum number of sub-polygons in the semi-infinite plate
    :return: pl_uc
    """
    inf_plates = pl_uc.get_plates_of_type("inf")
    if typ == "rational":
        plate_to_hold = deepcopy(inf_plates[np.argmin([len(plate) for plate in inf_plates])])
        semi_inf_plate = sample_semi_inf_rational_plate(plate_to_hold, num_plates=num_plates, position=position)
    else:
        semi_inf_plate = sample_semi_inf_plate_along_uc(pl_uc, d_min=d_min, max_plates=max_plates)
    pl_uc.add_plate(semi_inf_plate)
    return pl_uc


def add_finite_plate(pl_uc, max_plates=5, tries=5):
    """
    Adds a random finite plate to the plate lattice unit cell.
    :param pl_uc: plate lattice unit cell.
    :param max_plates: maximum number of subpolygons in the finite plate.
    :param tries: maximum number of tries to get a finite plate.
    :return: pl_uc with new finite plate.
    """
    finite_plate = sample_finite_plate(pl_uc, max_plates, tries)
    if finite_plate is None:
        warnings.warn("Could not find a finite plate to add to the unit cell in {} tries".format(tries))
        return pl_uc
    else:
        pl_uc.add_plate(finite_plate)
        return pl_uc

def get_poly_lines(polygon: Polygon, intersection_list: IntersectionList = None, ix=None):
    """
    Returns the lines of a polygon as a PolygonLineCollection.
    :param polygon:
    :param intersection_list:
    :param ix:
    :return:
    """
    poly_lines = PolygonLineCollection.from_Polygon(polygon)
    poly_id = polygon.id if ix is None else ix
    ints_poly, line_ixs = intersection_list.get_polygon_intersections(poly_id)
    ints_poly.add_middle_intersection_points()

    for int_line in ints_poly:
        if len(int_line.line_pts) > 1:
            poly_lines.add_intersection_line(PolygonLine.from_Intersection(int_line))
    return poly_lines


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


def get_edge_periodicity(edge: PolygonNodeEdge, epsilon=epsilon):
    """
    Returns the periodic copies of an edge.
    :param edge:
    :param epsilon:
    :return:
    """
    os_bd = {0.0: 1.0, 1.0: 0.0}
    edge = deepcopy(edge)
    periodic_edges = [edge]
    edge_line = Line.from_points(*edge.points())
    for plane in uc:
        if edge_line.direction.is_parallel(plane.normal):
            continue
        if plane.project_line(edge_line).is_close(edge_line, abs_tol=epsilon):
            new_edge = deepcopy(edge)
            for p in new_edge.points():
                p[[np.argwhere(plane.normal)[0]][0]] = os_bd[plane.point[np.argwhere(plane.normal)[0]][0]]
            periodic_edges.append(new_edge)

    if len(periodic_edges) == 3:
        edge = deepcopy(periodic_edges[1])
        for plane in uc:
            if edge_line.direction.is_parallel(plane.normal):
                continue
            if plane.project_line(edge_line).is_close(edge_line, abs_tol=epsilon):
                new_edge = deepcopy(edge)
                for p in new_edge.points():
                    p[[np.argwhere(plane.normal)[0]][0]] = os_bd[plane.point[np.argwhere(plane.normal)[0]][0]]
                if not line_in_line_list(Line.from_points(*new_edge.points()),
                                         [Line.from_points(*l.points()) for l in periodic_edges]):
                    periodic_edges.append(new_edge)

    return periodic_edges


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


def is_valid_sub_poly(sub_poly, poly_lines):
    """
    Checks if a sub-polygon is valid. A sub-polygon is valid if all of its
    edges are either intersection lines or edge lines that are periodic.
    :param sub_poly:
    :param poly_lines:
    :return:
    """
    valid = True
    edges_outside = []

    for i in range(len(sub_poly)):
        i1 = 0 if i + 1 == len(sub_poly) else i + 1
        edge = poly_lines.get_edge([sub_poly[i], sub_poly[i1]])
        if not edge.is_intersection and edge.is_edge:
            edges_outside.append(edge)

    for edge in edges_outside:
        if not any([edge.coords_equal(e) for e in
                    get_edge_periodicity(edges_outside[0])]) or len(edges_outside) == 1:
            valid = False

    return valid


def same_edges(sub_poly1, sub_poly2):
    """
    Checks if two sub-polygons are connected by at least one edge that is the same.
    :param sub_poly1:
    :param sub_poly2:
    :return:
    """
    edges_outside1 = []
    for i in range(len(sub_poly1[1])):
        i1 = 0 if i + 1 == len(sub_poly1[1]) else i + 1
        edge = sub_poly1[0].get_edge([sub_poly1[1][i], sub_poly1[1][i1]])
        if not edge.is_intersection and edge.is_edge:
            edges_outside1.append(edge)

    edges_outside2 = []
    for i in range(len(sub_poly2[1])):
        i1 = 0 if i + 1 == len(sub_poly2[1]) else i + 1
        edge = sub_poly2[0].get_edge([sub_poly2[1][i], sub_poly2[1][i1]])
        if not edge.is_intersection and edge.is_edge:
            edges_outside2.append(edge)

    for e1 in edges_outside1:
        for e2 in edges_outside2:
            for l in get_edge_periodicity(e2):
                if l.coords_equal(e1):
                    return True

    return False


def joined_polygon_indexes(sub_polys_to_join):
    """
    Returns the indexes of the sub-polygons that are connected.
    :param sub_polys_to_join:
    :return:
    """
    join_ixs = []
    for i, sub_poly1 in enumerate(sub_polys_to_join):
        for j, sub_poly2 in enumerate(sub_polys_to_join[i + 1:]):
            if same_edges(sub_poly1, sub_poly2):
                join_ixs.append([i, j + i + 1])
    return join_ixs


def get_connected(join_ixs):
    """
    Checks which sub-polygons are connected to each other.
    :param join_ixs:
    :return:
    """
    connected_ixs = []
    used_ixs = []
    for i in range(len(join_ixs)):
        if i in used_ixs:
            continue
        next_ix = 0
        connected = join_ixs[i]
        while next_ix==0:
            added = 0
            for i in set(range(len(join_ixs)))-set(used_ixs):
                if join_ixs[i][0] in connected or join_ixs[i][1] in connected:
                    connected += join_ixs[i]
                    used_ixs.append(i)
                    added = 1
            if added == 0:
                next_ix = 1
        connected_ixs.append(list(set(connected)))

    return connected_ixs


def get_plate_valid_sub_polys(pl_uc: PlateList, intersection_list: IntersectionList, id_dict=None):
    """
    Returns the valid sub-polygons of a plane in a random normal.
    :param pl_uc: plate lattice it will intersect with
    :param intersection_list: intersections of the plane in the unit cell with the plate lattice
    :param id_dict: dictionary of polygon ids, they can have different ones if the intersection list is done with
    the nearest neighbors
    :return: valid_sub_polys_plate
    """
    valid_sub_polys_plate = {}
    for plate in pl_uc:
        sub_polys_to_join = []
        valid_sub_polys = []
        for poly in plate:
            if len(poly.vertices) >= 3:
                ix = poly.id if id_dict is None else id_dict[poly.id]
                polylines = get_poly_lines(poly, intersection_list, ix)
                sub_polys = get_sub_polys(polylines)
                for sub_poly in sub_polys:
                    if is_valid_sub_poly(sub_poly, polylines):
                        valid_sub_polys.append([(deepcopy(polylines), sub_poly)])
                    else:
                        sub_polys_to_join.append((deepcopy(polylines), sub_poly))

        join_ixs = joined_polygon_indexes(sub_polys_to_join)
        for ixs in get_connected(join_ixs):
            sub_poly = [sub_polys_to_join[i] for i in ixs]
            valid_sub_polys.append(sub_poly)

        valid_sub_polys_plate[plate.id] = valid_sub_polys
    return valid_sub_polys_plate


def get_sub_polygons_and_lines_for_graph(valid_sub_polys_plate,
                                         pl_uc: PlateList,
                                         intersection_list: IntersectionList):
    """
    Returns the valid sub-polygons and polyline collection of a unit cell to create a graph from it.
    :param valid_sub_polys_plate: valid sub-polygons of a plane in a random normal to form finite plates.
    :param pl_uc: plate lattice it will intersect with
    :param intersection_list: intersections of the plane in the unit cell with the plate lattice
    :return: all_sub_polygons, all_lines_in_sub_polygons
    """
    all_sub_polygons = []
    all_lines_in_sub_polygons = []
    ix = 0
    for plate in valid_sub_polys_plate:
        sub_polygon = dict()
        sub_polygon["id"] = 0
        sub_polygon["normal"] = copy(pl_uc[plate].__dict__["normal"])
        sub_polygon["type"] = copy(pl_uc[plate].__dict__["typ"])
        for valid_sub_polys in valid_sub_polys_plate[plate]:
            sub_polygon["id"] = ix
            ix += 1
            vertices = []
            lines_in_sub_poly = []

            for valid_sub_poly in valid_sub_polys:
                vxs = []
                nodes = valid_sub_poly[0].get_graph().nodes
                for n in valid_sub_poly[1]:
                    vxs.append(nodes[n]["point"])
                vertices.append(np.array(vxs))

                for i in range(len(valid_sub_poly[1])):
                    i1 = 0 if i + 1 == len(valid_sub_poly[1]) else i + 1
                    edge = valid_sub_poly[0].get_edge([valid_sub_poly[1][i], valid_sub_poly[1][i1]])
                    polyline = valid_sub_poly[0][edge.line_id]
                    line_id = intersection_list.find_line_from_PolyLine(polyline)
                    if line_id is not None and line_id not in lines_in_sub_poly:
                        lines_in_sub_poly.append(line_id)

            sub_polygon["polygon_vertices"] = vertices
            all_sub_polygons.append(deepcopy(sub_polygon))
            all_lines_in_sub_polygons.append(deepcopy(lines_in_sub_poly))

    return all_sub_polygons, all_lines_in_sub_polygons


def collapse_same_lines(line_ixs, intersection_list: IntersectionList):
    """
    Collapses the same lines in a list of line indexes.
    :param line_ixs:
    :param intersection_list:
    :return:
    """
    same_line_dict = {}
    used_line_ixs = []
    for i, line_id in enumerate(line_ixs):
        if i not in used_line_ixs:
            line = copy(intersection_list[line_id].line)
            l_periodicity = get_line_periodicity(line)
            same_line_dict[line_id] = []
            used_line_ixs.append(i)
            next_line = 0
            while next_line == 0:
                added = 0
                for j in set(range(len(line_ixs))) - set(used_line_ixs):
                    other_i = line_ixs[j]
                    other_line = copy(intersection_list[other_i].line)
                    if line_in_line_list(other_line, l_periodicity):
                        same_line_dict[line_id] += [other_i]
                        used_line_ixs.append(j)
                        other_periodicity = get_line_periodicity(other_line)
                        other_periodicity = [l for l in other_periodicity if not line_in_line_list(l, l_periodicity)]
                        l_periodicity = l_periodicity + other_periodicity
                        added = 1
                if added == 0:
                    next_line = 1

    reverse_same_line_dict = {}
    for key, lines in same_line_dict.items():
        for line_id in lines:
            reverse_same_line_dict[line_id] = key

    return reverse_same_line_dict


def get_graph(all_sub_polygons, all_lines_in_sub_polygons, intersection_list: IntersectionList):
    line_id_dict = {}
    unique_lines = list(set([j for i in all_lines_in_sub_polygons for j in i]))
    same_line_dict = collapse_same_lines(unique_lines, intersection_list)
    G = nx.MultiGraph()
    i = 0
    for line_id in unique_lines:
        node_ix = i if line_id not in same_line_dict.keys() else line_id_dict[same_line_dict[line_id]]
        if node_ix not in G.nodes:
            G.add_node(node_ix)
            G.nodes[node_ix]['line'] = [copy(intersection_list[line_id].line)]
            G.nodes[node_ix]['line_pts'] = [copy(intersection_list[line_id].line_pts)]
            G.nodes[node_ix]['plates'] = copy(intersection_list[line_id].plates)
            i += 1

        else:
            G.nodes[node_ix]['line'] += [copy(intersection_list[line_id].line)]
            G.nodes[node_ix]['line_pts'] += [copy(intersection_list[line_id].line_pts)]
            plates = copy(intersection_list[line_id].plates)
            for p in plates:
                if p not in G.nodes[node_ix]['plates']:
                    G.nodes[node_ix]['plates'] = G.nodes[node_ix]['plates'] + [p]
        line_id_dict[line_id] = node_ix

    for i, lines_sub_poly in enumerate(all_lines_in_sub_polygons):
        for j in range(len(lines_sub_poly)):
            lines_sub_poly[j] = same_line_dict[lines_sub_poly[j]] if lines_sub_poly[j] in same_line_dict.keys() \
                else lines_sub_poly[j]
        all_lines_in_sub_polygons[i] = list(set(lines_sub_poly))

    for node_ix, (sub_poly, line_ids) in enumerate(zip(all_sub_polygons, all_lines_in_sub_polygons)):
        for i, id1 in enumerate(line_ids):
            if len(line_ids) == 1:
                G.add_edges_from([(line_id_dict[id1], line_id_dict[id1], sub_poly)])
            for id2 in line_ids[i + 1:]:
                if not (line_id_dict[id1], line_id_dict[id2], node_ix) in G.edges:
                    G.add_edges_from([(line_id_dict[id1], line_id_dict[id2], sub_poly)])

    return G


def get_intersection_line_pts_from_poly_lines(poly_lines):
    intersection_lines = []
    for edge in poly_lines.all_edges():
        if edge.is_intersection:
            intersection_lines.append(np.array(edge.points()))
    return intersection_lines