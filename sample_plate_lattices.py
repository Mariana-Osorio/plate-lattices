from Plate_Lattice_UC import *
from plate_lattices import *


def sample_inf_plate(max_plates=5, max_int=4, position="random"):
    while True:
        normal = sample_normal_for_inf_plate(max_int)
        p = np.around(np.random.rand(3), 1) if position != "bfcc" else get_random_point_bfcc()
        lines_uc, vertices = intersect_plane_uc(Plane(point=p, normal=normal))
        lines = get_periodic_lines(vertices)
        polygons = get_periodic_polygons(normal, vertices, lines)
        polygons = clean_polys(polygons)
        if len(polygons) <= max_plates:
            break
    return normal, polygons


def base_plate_lattice(max_plates=5, max_int=4, position="random"):
    plate_lattice_uc = PlateLatticeUC()
    for i in range(2):
        while True:
            normal, polygons = sample_inf_plate(max_plates, max_int, position=position)
            if not plate_lattice_uc.has_normal(normal):
                break
        plate_lattice_uc.add_to_plate_lattice_uc(normal, polygons)

    return plate_lattice_uc


def distr_plates(n_plates):
    options = [l for l in list(product(range(n_plates + 1), repeat=3)) if sum(l) == n_plates]
    option = options[np.random.choice(range(len(options)))]
    return option


def new_inf_plate(plate_lattice_uc, max_plates=5, max_int=4, position="random"):
    while True:
        normal, polygons = sample_inf_plate(max_plates=max_plates, max_int=max_int, position=position)
        if len(polygons) == 0 or len(polygons[0]) == 0:
            continue
        if not plate_lattice_uc.has_plane(Plane(normal=normal, point=polygons[0][0]), epsilon=1e-06):
            break
    return normal, polygons


def sample_semi_inf_plate(plate_lattice_uc, d_min=0.3, max_plates=5):
    if plate_lattice_uc.nn_polygons is None:
        plate_lattice_uc.nn_uc()
    plane = random_plane_for_semi_inf(plate_lattice_uc.nn_normals)
    valid_plates = [i for i, n in enumerate(plate_lattice_uc.nn_normals) if n[plane] == 0]
    valid_poly_ixs = [i for i, idx in enumerate(plate_lattice_uc.nn_plane_id) if idx in valid_plates]
    valid_polys = [plate_lattice_uc.nn_polygons[i] for i in valid_poly_ixs]
    lines_for_semi_inf = plate_lattice_uc.get_uc_lines(valid_polys)[['yz', 'xz', 'xy'][plane]]
    # semi-infinite polygon
    while True:
        semi_inf_lines_pts = get_semi_inf_line_pts(lines_for_semi_inf, d_min)
        normal, semi_inf_polys = get_semi_inf_polys(plane, semi_inf_lines_pts)
        if len(semi_inf_polys) <= max_plates:
            break

    return normal, semi_inf_polys


def new_semi_inf_plate(plate_lattice_uc, d_min=0.3, max_plates=5):
    while True:
        normal, polygons = sample_semi_inf_plate(plate_lattice_uc, d_min=d_min, max_plates=max_plates)
        if not plate_lattice_uc.has_plane(Plane(normal=normal, point=polygons[0][0])):
            break
    return normal, polygons


def sample_finite_plate(plate_lattice_uc):
    inside_sub_polygons = []
    i = 0
    while len(inside_sub_polygons) < 1:
        if i > 4:
            return None, None, None
        # Sample plate
        plane, edge_lines, poly = sample_random_plate()
        # Get polygon intersections
        polygon_intersections = get_poly_intersections(poly, plane, edge_lines, plate_lattice_uc)

        # Get sub polygon cycles
        nodes = polygon_intersections.all_points.to_dict()
        sub_polys = get_sub_polygon_cycles(polygon_intersections)
        lines_inside = [l.intersection for l in polygon_intersections.lines]
        inside_sub_polygons = get_inside_sub_polygons(sub_polys, lines_inside, min_area=0.05, nodes=nodes)
        i += 1

    # Choose random inside subpolygon
    j = np.random.choice(range(len(inside_sub_polygons)))
    poly_ixs = inside_sub_polygons[j][0]
    lines_in_poly = inside_sub_polygons[j][1]
    poly = np.array([nodes[i] for i in poly_ixs])
    edge_lines = [polygon_intersections.lines[i].line for i in lines_in_poly]
    polygon_intersections = get_poly_intersections(poly, plane, edge_lines, plate_lattice_uc)

    return poly, plane, polygon_intersections


def get_poly_intersections(polygon, plane, edge_lines, plate_lattice_uc):
    plate_intersections = PlateIntersections.intersect_w_polygon(plate_lattice_uc, polygon, plane)
    plate_intersections.add_intersection_points()
    polygon_intersections = PolygonIntersections(polygon, plane, edge_lines)
    polygon_intersections.add_intersection_lines(plate_intersections)
    polygon_intersections.order_lines()
    return polygon_intersections


def get_next_poly_cycles(plane, pts_to_match, plate_lattice_uc):
    for i in range(3):
        for val in [0.0, 1.0]:
            where = where_pt_equal(pts_to_match[:, i], val)
            if len(where) == len(pts_to_match[:, i]):
                p = i
                value = [v for v in [0.0, 1.0] if v != val][0]

    pts_to_match[:, p] = value
    plane_to_match = Plane(normal=plane.normal, point=pts_to_match[0])
    edge_lines, polygon = intersect_plane_uc(plane_to_match)
    poly_ints = get_poly_intersections(polygon, plane_to_match, edge_lines, plate_lattice_uc)
    wanted_nodes = [poly_ints.all_points.get_point_id(PointInLine(pt_to_match, 0)) for pt_to_match in pts_to_match]
    wanted_line_ix = [i for i in range(len(poly_ints)) if
                      set(poly_ints.lines[i].point_ids).intersection(set(wanted_nodes)) == set(wanted_nodes)]
    sub_poly_cycles = [[sp, sp_lines] for (sp, sp_lines) in get_sub_polygon_cycles(poly_ints) if
                       set(sp).intersection(set(wanted_nodes)) == set(wanted_nodes)]

    valid_sub_poly_cycles = []
    for (cyc, cyc_lines) in sub_poly_cycles:
        num_edges_outside = sum([1 for i in cyc_lines if not poly_ints.lines[i].intersection and i
                                 != wanted_line_ix[0]])
        if num_edges_outside <= 1:
            valid_sub_poly_cycles.append([cyc, cyc_lines])
    return valid_sub_poly_cycles, poly_ints


def get_next_poly(plane, pts_to_match, plate_lattice_uc):
    sub_poly_cycles, poly_ints = get_next_poly_cycles(plane, pts_to_match, plate_lattice_uc)
    if len(sub_poly_cycles) == 0:
        return None, None
    else:
        j = np.random.choice(range(len(sub_poly_cycles)))
        poly_ixs = sub_poly_cycles[j][0]
        lines_in_poly = sub_poly_cycles[j][1]
        poly = np.array([poly_ints.all_points.to_dict()[i] for i in poly_ixs])
        edge_lines = [poly_ints.lines[i].line for i in lines_in_poly]
    return poly, edge_lines


def sample_finite_polygons(plate_lattice_uc, max_plates=5, verbose=False):
    i = 0
    while True:
        finite_polygons = []
        polygon, plane, polygon_intersections = sample_finite_plate(plate_lattice_uc)

        if polygon is not None:
            normal = np.array(plane.normal)
            finite_polygons.append(polygon)
            pts_to_match = [polygon_intersections.lines[i].points.copy() for i in range(len(polygon_intersections)) if
                            not polygon_intersections.lines[i].intersection]
            j = 0
            while len(pts_to_match) > 0:
                pts_to_match = pts_to_match[0]
                polygon, edge_lines = get_next_poly(plane, pts_to_match, plate_lattice_uc)
                if polygon is None:
                    break
                finite_polygons.append(polygon)
                plane = Plane(normal=normal, point=polygon[0])
                polygon_intersections = get_poly_intersections(polygon, plane,
                                                               edge_lines, plate_lattice_uc)
                pts_to_match = [polygon_intersections.lines[i].points.copy() for i in range(len(polygon_intersections))
                                if
                                not polygon_intersections.lines[i].intersection and
                                not arrays_equal(polygon_intersections.lines[i].points, pts_to_match)]

                if j >= max_plates - 2:
                    print_verbose(verbose, "Exceeded plate number with matching.")
                    polygon = None
                    break
                j += 1

            if polygon is not None:
                break

        if i > 9:
            raise Exception("No finite plates.")
        i += 1

    return normal, finite_polygons


def print_verbose(verbose, text):
    if verbose:
        print(text)


def new_finite_plate(plate_lattice_uc, max_plates=5, verbose=False):
    sample_again = 0
    while True:
        try:
            normal, polygons = sample_finite_polygons(plate_lattice_uc, max_plates=max_plates, verbose=verbose)
        except Exception as e:
            print_verbose(verbose, f"Ran into '{e.__str__()}'.")
            print_verbose(verbose, f"Resampling.")
            sample_again = 1
            return None, None, sample_again

        if not plate_lattice_uc.has_plane(Plane(normal=normal, point=polygons[0][0]), epsilon=1e-06):
            break
    return normal, polygons, sample_again


def sample_plate_lattice(n_inf_p, n_semi_inf_p, n_finite_p, max_int=4, position="bfcc", max_plates=3, d_min=0.3,
                         verbose=False):
    while True:
        sample_again = 0
        plate_lattice_uc = base_plate_lattice(max_plates=max_plates, max_int=max_int, position=position)
        print_verbose(verbose, f"Base plate lattice created, normals: {plate_lattice_uc.normals}")

        for i in range(n_inf_p):
            normal, polygons = new_inf_plate(plate_lattice_uc, max_plates=max_plates, max_int=max_int,
                                             position=position)
            plate_lattice_uc.add_to_plate_lattice_uc(normal, polygons)
            print_verbose(verbose, f"Infinite plate {i + 1} created, {len(polygons)} polygons.")

        for i in range(n_semi_inf_p):
            normal, semi_inf_polys = new_semi_inf_plate(plate_lattice_uc, d_min=d_min, max_plates=max_plates)
            plate_lattice_uc.add_to_plate_lattice_uc(normal, semi_inf_polys)
            print_verbose(verbose, f"Semi-infinite plate {i + 1} created, {len(semi_inf_polys)} polygons.")

        for i in range(n_finite_p):
            normal, finite_polygons, sample_again = new_finite_plate(plate_lattice_uc, max_plates=max_plates,
                                                                     verbose=verbose)
            if sample_again != 0:
                break
            plate_lattice_uc.add_to_plate_lattice_uc(normal, finite_polygons)
            print_verbose(verbose, f"Finite plate {i + 1} created, {len(finite_polygons)} polygons.")

        if sample_again == 0:
            print_verbose(verbose, f"Finished.")
            break

    return plate_lattice_uc


def random_plate_lattice(n_plates, max_int=4, position="bfcc", max_plates=3, d_min=0.3, verbose=False):
    assert n_plates >= 2, "Plate lattice must have at least 2 sets of plates."
    n_inf_p, n_semi_inf_p, n_finite_p = distr_plates(n_plates - 2)
    plate_lattice_uc = sample_plate_lattice(n_inf_p, n_semi_inf_p, n_finite_p, max_int=max_int,
                                            position=position, max_plates=max_plates, d_min=d_min, verbose=verbose)

    return plate_lattice_uc
