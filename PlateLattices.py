from skspatial.typing import array_like
from pl_functions import *


class Polygon:
    def __init__(self, vertices, edge_lines, normal=None, id=0):
        assert all(isinstance(x, Line) for x in edge_lines), "Not all edge lines are Line objects."
        self.edge_lines = edge_lines
        self.vertices = vertices
        self.normal = normal if normal is not None else np.cross(vertices[0] - vertices[1], vertices[0] - vertices[2])
        self.plane = get_plane_from_poly(vertices, self.normal)
        self.id = id
        self.all_lines = []

    def __repr__(self):
        return f"Polygon({len(self.vertices)} vertices)"

    def __len__(self):
        return len(self.vertices)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class PolygonList:
    def __init__(self, polygons: list, new_ids=True):
        assert all(isinstance(x, Polygon) for x in polygons), "Not all polygons are Polygon objects."
        polys = []
        for p in polygons:
            polys.append(deepcopy(p))

        self.polygons = polys
        if new_ids:
            for i in range(len(self.polygons)):
                self.polygons[i].id = i

    @classmethod
    def from_lists(cls, vertices: list, edge_lines: list, normal=None):
        vertices, ixs = clean_polys(vertices)
        if len(vertices) == 0:
            raise ValueError("No polygons left after cleaning.")
        edge_lines = [edge_lines[i] for i in ixs]
        polygons = [Polygon(vertices[i], edge_lines[i], normal) for i in range(len(vertices))]
        return cls(polygons)

    def __repr__(self):
        return f"PolygonList({len(self.polygons)} polygons)"

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, idx):
        return self.polygons[idx]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def vert_list(self):
        return [p.vertices for p in self.polygons]

    def plane_list(self):
        return [p.plane for p in self.polygons]

    def edge_line_list(self):
        return [p.edge_lines for p in self.polygons]

    def ids(self):
        return [p.id for p in self.polygons]

    def add_polygon(self, polygon: Polygon, new_id=True):
        assert isinstance(polygon, Polygon), "Not a Polygon object."
        if new_id:
            polygon.id = len(self.polygons)
        self.polygons.append(polygon)


class Plate:
    def __init__(self, normal: array_like, polygons: PolygonList, typ: str, id=0):
        self.id = id
        self.normal = normal
        self.polygons = polygons
        self.typ = typ

    def __repr__(self):
        return f"Plate({self.typ}, {len(self.polygons)} polygons)"

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, idx):
        return self.polygons[idx]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def vert_list(self):
        return self.polygons.vert_list()

    def plane_list(self):
        return self.polygons.plane_list()

    def polygon_ids(self):
        return self.polygons.ids()

    def idx_polygon(self, ix):
        i = self.polygons.ids().index(ix)
        return self.polygons[i]

    def change_poly_ids(self, new_ids):
        assert len(new_ids) == len(self.polygons), "New ids must be same length as polygons."
        assert all([type(x) == int for x in new_ids]), "New ids must be integers."
        for i in range(len(self.polygons)):
            self.polygons[i].id = new_ids[i]

    def get_neighbor(self, idx):
        assert type(idx) == tuple and len(idx) == 3 and type(idx[0]) == int \
               and type(idx[1]) == int and type(idx[2]) == int, "Not valid idx"
        neighbor = deepcopy(self)
        for poly in neighbor:
            poly.edge_lines = [Line(point=l.point + idx, direction=l.direction) for l in poly.edge_lines]
            poly.vertices = poly.vertices + idx
            poly.plane = get_plane_from_poly(poly.vertices, poly.normal)

        return neighbor

    def has_plane(self, plane):
        return plane_in_plane_list(plane, self.plane_list())

    def get_nn_uc_plate(self):
        plate_nn = deepcopy(self)
        for idx in product([0, 1, -1], [0, 1, -1], [0, 1, -1]):
            if idx != (0, 0, 0):
                neighbor = self.get_neighbor(idx)
                for poly in neighbor:
                    if plate_nn.typ != "inf" or not plate_nn.has_plane(poly.plane) and plate_nn.typ == "inf":
                        edge_lines, vertices = get_vertices_and_lines_in_uc(deepcopy(poly))
                        if len(vertices) != 0:
                            poly.edge_lines = edge_lines
                            poly.vertices = vertices
                            plate_nn.polygons.add_polygon(poly)
        return plate_nn


class PlateList:
    def __init__(self, plates: list, new_ids=True):
        assert all(isinstance(x, Plate) for x in plates), "Not all plates are Plate objects."
        plts = []
        for p in plates:
            plate_lst = []
            for plate in plts:
                plate_lst += plate.plane_list()
            if not plane_in_plane_list(p[0].plane, plate_lst):
                plts.append(deepcopy(p))

        self.plates = plts

        if new_ids:
            j = 0
            for i in range(len(self.plates)):
                self[i].id = i
                for p in self[i]:
                    p.id = j
                    j += 1

    @classmethod
    def from_lists(cls, normal_list, polygons_list, polygon_edges_list, plate_type_list):
        plates = []

        for i in range(len(normal_list)):
            polygons = []
            for j in range(len(polygons_list[i])):
                polygons.append(Polygon(polygons_list[i][j], polygon_edges_list[i][j], normal_list[i]))

            plates.append(Plate(normal_list[i], PolygonList(polygons), plate_type_list[i]))

        return cls(plates)

    def __repr__(self):
        return f"PlateList({len(self.plates)} plates)"

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, idx):
        return self.plates[idx]

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __iter__(self):
        return iter(self.plates)

    def get_plates_of_type(self, typ):
        return PlateList([p for p in self if p.typ == typ])

    def polygon_list(self):
        poly_list = []
        for p in self.plates:
            for poly in p.polygons:
                poly_list.append(poly)
        return PolygonList(poly_list)

    def vert_list(self):
        vert_list = []
        for p in self:
            vert_list += p.vert_list()
        return vert_list

    def plane_list(self):
        plane_list = []
        for p in self:
            plane_list += p.plane_list()
        return plane_list

    def normal_list(self):
        normal_list = []
        for p in self:
            normal_list.append(p.normal)
        return normal_list

    def plate_ids(self):
        return [p.id for p in self for _ in p]

    def polygon_ids(self):
        polygon_ids = []
        for p in self:
            polygon_ids += p.polygon_ids()
        return polygon_ids

    def idx_polygon(self, ix):
        ixs = self.polygon_ids()
        i = ixs.index(ix)
        return self.polygon_list()[i]

    def add_plate(self, plate, new_ids=True):
        assert isinstance(plate, Plate), "Plate must be Plate object."

        if not self.has_plane(plate[0].plane):
            i = len(self.plates)
            j = len(self.polygon_list())
            self.plates.append(deepcopy(plate))
            if new_ids:
                self[i].id = i
            for p in self[i]:
                p.id = j
                j += 1

    def has_plane(self, plane, epsilon=1e-10):
        return True if plane_in_plane_list(plane, self.plane_list(), epsilon=epsilon) else False

    def has_normal(self, normal, epsilon=1e-10):
        dup_normals = [1 for n in self.normal_list() if
                       Vector(n).is_parallel(normal, abs_tol=epsilon)]
        return True if len(dup_normals) != 0 else False

    def get_neighbor(self, idx):
        assert type(idx) == tuple and len(idx) == 3 and type(idx[0]) == int \
               and type(idx[1]) == int and type(idx[2]) == int, "Not valid idx"
        neighbor = []
        for plate in self:
            plate = deepcopy(plate)
            for poly in plate:
                poly.edge_lines = [Line(point=l.point + idx, direction=l.direction) for l in poly.edge_lines]
                poly.vertices = poly.vertices + idx
                poly.plane = get_plane_from_poly(poly.vertices, poly.normal)
            neighbor.append(plate)

        neighbor = PlateList(neighbor)
        return neighbor

    def get_nn_uc_plates(self, return_dict=False):
        plates_nn = PlateList([])
        plates_nn_dict = {}
        for plate in self:
            plate_nn = plate.get_nn_uc_plate()
            i = 0
            for poly in plate:
                plates_nn_dict[poly.id] = len(plates_nn.polygon_ids()) + i
                i += 1
            plates_nn.add_plate(plate_nn)
        if return_dict:
            return plates_nn, plates_nn_dict
        else:
            return plates_nn

    def get_plate_info(self):
        normal_list, polygons_list, polygon_edges_list, plate_type_list = [], [], [], []
        for plate in self:
            normal_list.append(plate.normal)
            polygons_list.append(plate.polygons.vert_list())
            polygon_edges_list.append(plate.polygons.edge_line_list())
            plate_type_list.append(plate.typ)
        return normal_list, polygons_list, polygon_edges_list, plate_type_list

    def get_uc_plane_lines(self, poly_ixs=None):
        lines = {'yz': [], 'xz': [], 'xy': []}
        poly_list = self.polygon_list() if poly_ixs is None else [self.polygon_list()[i] for i in poly_ixs]
        for poly in poly_list:
            for val in [0., 1.]:
                for plane in range(3):
                    if len(poly.vertices[poly.vertices[:, plane] == val]) >= 2:
                        line = poly.vertices[poly.vertices[:, plane] == val][:, [i for i in [0, 1, 2] if i != plane]]
                        if not array_in_list(line, lines[list(lines.keys())[plane]]):
                            lines[list(lines.keys())[plane]] += [line]
        return lines


class Node:
    def __init__(self, point: np.ndarray, id=0):
        assert type(point) in [np.ndarray, Point], "Must be an array or point."
        assert len(point) == 3, "Must be a 3D point."
        self.point = np.array(point)
        self.id = id

    def __repr__(self):
        return f"Node({np.around(self.point, 3)})"


class PolygonNodeEdge:
    def __init__(self, nodes: list, line_id: int, is_edge: bool = False,
                 is_intersection: bool = False):
        assert type(nodes) == list, "Must be a list"
        assert len(nodes) == 2, "Edge must connect only 2 nodes."
        self.nodes = nodes
        self.line_id = line_id
        self.is_edge = is_edge
        self.is_intersection = is_intersection

    def __repr__(self):
        return f"PolygonNodeEdge({self.pt_ids()}, " \
               f"intersection={self.is_intersection}, edge={self.is_edge})"

    def coords_equal(self, other):
        return all([point_in_list(p, self.points()) for p in other.points()])

    def points(self):
        return [self.nodes[0].point, self.nodes[1].point]

    def pt_ids(self):
        return [self.nodes[0].id, self.nodes[1].id]


class Intersection:
    def __init__(self, line: Line, line_pts: list, plates: list, polygons: list):
        self.line = line
        self.line_pts = line_pts
        self.pt_ids = list(range(len(line_pts)))
        self.plates = plates
        self.polygons = polygons

    def __repr__(self):
        return f"Intersection({self.plates} planes)"

    def __len__(self):
        return len(self.line_pts)

    def __getitem__(self, idx):
        return self.line_pts[idx]

    def __eq__(self, other):
        if self.line.is_close(other.line):
            if arrays_equal(np.array(self.line_pts), np.array(other.line_pts)):
                return True
        return False

    def order_line_pts(self):
        order = arg_order_points_on_line(self.line_pts, self.line.direction)
        self.line_pts = np.array([self.line_pts[i] for i in order])
        self.pt_ids = [self.pt_ids[i] for i in order]

    def find_pt(self, point):
        where = point_in_list(point, self.line_pts, return_where=True)
        if len(where) == 0:
            return None
        else:
            return where[0]

    def get_pt_id(self, point):
        where = point_in_list(point, self.line_pts, return_where=True)
        if len(where) == 0:
            return None
        else:
            return self.pt_ids[where[0]]

    def add_point(self, point):
        where = self.find_pt(point)
        if where is None:
            pt_id = len(self.line_pts)
            self.line_pts = np.vstack((self.line_pts, point))
            self.pt_ids.append(pt_id)
            self.order_line_pts()
            return pt_id
        else:
            return self.pt_ids[where]


class PolygonLine:
    def __init__(self, line: Line, line_pts: list,
                 are_edges: list = [], are_intersections: list = [],
                 pt_ids: list = [], id=0):
        assert type(line) == Line, "Must be a Line object."
        assert all([type(p) in [np.ndarray, Point] for p in line_pts]), \
            "Must be a list of arrays or points."
        o1 = list(range(len(line_pts)))
        o2 = list(range(len(line_pts)))
        o2.reverse()
        order = list(arg_order_points_on_line(line_pts, line.direction))
        assert order == o1 or order == o2, "Points must be ordered along the line."
        if pt_ids != []:
            assert len(pt_ids) == len(line_pts), \
                "Must be a list of ints, one for each line point."
        are_edges = [False] * len(line_pts) if are_edges == [] else are_edges
        are_intersections = [False] * len(line_pts) if \
            are_intersections == [] else are_intersections
        assert len(are_edges) == len(line_pts) - 1, \
            "Must be a list of bools, one less than the number of line points."
        assert len(are_intersections) == len(line_pts) - 1, \
            "Must be a list of bools, one less than the number of line points."
        self.line = line
        self.line_pts = [Node(p, i) for i, p in enumerate(line_pts)] if \
            pt_ids == [] else [Node(p, i) for i, p in zip(pt_ids, line_pts)]
        self.order_line_pts()
        self.id = id
        self.edges = []
        self.create_edges(are_edges, are_intersections)

    @classmethod
    def from_Intersection(cls, intersection: Intersection, pt_ids=[], id=0):
        are_edges = [False] * (len(intersection) - 1)
        are_intersections = [True] * (len(intersection) - 1)
        return cls(intersection.line, intersection.line_pts, are_edges, are_intersections,
                   pt_ids, id)

    def __repr__(self):
        return f"PolygonLine({len(self.line_pts)} pts)"

    def __len__(self):
        return len(self.line_pts)

    def __getitem__(self, item):
        return self.line_pts[item]

    def __iter__(self):
        yield from self.line_pts

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def change_line_id(self, new_id):
        self.id = new_id
        self.update_edge_ids()

    def get_previous_edge_vals(self, pt_id):
        previous_edge = []
        for i in range(len(self.line_pts) - 1):
            if pt_id == self[i].id:
                previous_edge.append(self[i + 1].id)
            if pt_id == self[i + 1].id:
                previous_edge.append(self[i].id)
        return self.get_edge_values(previous_edge)

    def get_edge(self, pt_ids):
        for edge in self.edges:
            if sorted(edge.pt_ids()) == sorted(pt_ids):
                return edge
        return None

    def update_edge_ids(self):
        for edge in self.edges:
            edge.line_id = self.id

    def get_edge_values(self, pt_ids):
        edge = self.get_edge(pt_ids)
        return edge.is_edge, edge.is_intersection

    def edge_pt_ids(self):
        return [edge.pt_ids() for edge in self.edges]

    def order_line_pts(self):
        order = arg_order_points_on_line(self.point_coords(), self.line.direction)
        self.line_pts = [self.line_pts[i] for i in order]

    def point_ids(self):
        return [p.id for p in self.line_pts]

    def point_coords(self):
        return [p.point for p in self.line_pts]

    def find_pt(self, point):
        where = point_in_list(point, self.point_coords(), return_where=True)
        if len(where) == 0:
            return None
        else:
            return where[0]

    def get_pt_id(self, point):
        where = point_in_list(point, self.point_coords(), return_where=True)
        if len(where) == 0:
            return None
        else:
            return self.point_ids()[where[0]]

    def point_in_poly_line(self, coords, epsilon=epsilon):
        line_segment = LineSegment(self[0].point, self[-1].point)
        if line_segment.contains_point(coords, abs_tol=epsilon):
            return True
        else:
            return False

    def add_edge(self, node_ids: list, is_edge: bool = False,
                 is_intersection: bool = False):
        assert node_ids[0].id in self.point_ids(), "Node 1 not in line."
        assert node_ids[1].id in self.point_ids(), "Node 2 not in line."
        self.edges.append(PolygonNodeEdge(node_ids, self.id, is_edge,
                                          is_intersection))

    def create_edges(self, are_edges: list = [], are_intersections: list = []):
        self.edges = []
        for i in range(len(self.line_pts) - 1):
            self.add_edge([self[i], self[i + 1]],
                          are_edges[i], are_intersections[i])

    def are_intersections(self):
        return [e.is_intersection for e in self.edges]

    def are_edges(self):
        return [e.is_edge for e in self.edges]

    def all_edge_node_ids(self):
        return [p.id for p in self]

    def change_pt_ids(self, new_ids: list):
        assert len(new_ids) == len(self.line_pts), \
            "Must be a list of ints, one for each line point."
        for i, p in enumerate(self.line_pts):
            p.id = new_ids[i]

    def add_point(self, point: Node):
        where = self.find_pt(point.point)
        if where is None:
            assert point.id not in self.point_ids(), "Point id taken."
            self.line_pts.append(point)
            self.order_line_pts()
            are_edges = []
            are_intersections = []
            prev_values = self.get_previous_edge_vals(point.id)
            for i in range(len(self.line_pts) - 1):
                if point.id in [self[i].id, self[i + 1].id]:
                    are_edges.append(prev_values[0])
                    are_intersections.append(prev_values[1])
                else:
                    edge_values = self.get_edge_values([self[i].id, self[i + 1].id])
                    are_edges.append(edge_values[0])
                    are_intersections.append(edge_values[1])
            self.create_edges(are_edges, are_intersections)

    def add_intersection_line(self, intersection_line):
        # Adds an intersection line to the current line. It is assumed that
        # the intersection line has the same Line as the current line and
        # that points in the intersection line have the right ids.
        assert type(intersection_line) == PolygonLine, "Must be a PolygonLine object."

        for point in intersection_line:
            if point.id not in self.point_ids():
                self.line_pts.append(point)

        self.order_line_pts()
        pt_ids_ordered = [i for i in self.point_ids() if
                          i in intersection_line.point_ids()]

        are_edges = [True] * (len(self.line_pts) - 1)
        are_intersections = []
        is_intersection = False
        for i in range(len(self.line_pts) - 1):
            if pt_ids_ordered[0] == self[i].id:
                is_intersection = True
            are_intersections.append(is_intersection)
            if pt_ids_ordered[-1] == self[i + 1].id:
                is_intersection = False
        self.create_edges(are_edges, are_intersections)


class IntersectionList:
    def __init__(self, intersections: list):
        assert type(intersections) == list, "Not valid intersections"
        assert all([type(i) == Intersection for i in intersections]), "Not valid intersections"
        self.intersections = []
        for i in range(len(intersections)):
            self.add_intersection(intersections[i])


    def __repr__(self):
        return f"IntersectionList({len(self.intersections)} intersections)"

    def __len__(self):
        return len(self.intersections)

    def __getitem__(self, idx):
        return self.intersections[idx]

    @classmethod
    def from_plate_list(cls, plate_list: PlateList):
        intersections = []
        plate_ids = plate_list.plate_ids()
        polygon_list = plate_list.polygon_list()
        for i in range(len(polygon_list)):
            for j in range(i + 1, len(polygon_list)):
                poly1 = polygon_list[i]
                poly2 = polygon_list[j]
                if poly1.plane.is_close(poly2.plane, abs_tol=epsilon):
                    continue
                line_intersection = intersect_2_planes(poly1.plane, poly2.plane)
                if line_intersection is not None:
                    line_points = line_in_polys(line_intersection, poly1, poly2)
                    if len(line_points) != 0:
                        intersection = Intersection(line_intersection, line_points, [plate_ids[i], plate_ids[j]],
                                                    [poly1.id, poly2.id])
                        intersections.append(intersection)
        return cls(intersections)

    def get_polygon_intersections(self, polygon_id):
        poly_list = []
        line_ixs = []
        for i, intersection in enumerate(self):
            if polygon_id in intersection.polygons:
                poly_list.append(deepcopy(intersection))
                line_ixs.append(i)

        return IntersectionList(poly_list), line_ixs

    def line_list(self):
        return [i.line for i in self]

    def line_pts_list(self):
        return [i.line_pts for i in self]

    def plates_list(self):
        return [i.plates for i in self]

    def polygons_list(self):
        return [i.polygons for i in self]

    def add_intersection(self, intersection):
        assert type(intersection) == Intersection, "Not valid intersection"
        where = line_in_line_list(intersection.line, self.line_list(), return_where=True)
        if len(where) == 0:
            self.intersections.append(intersection)
        elif len(where) != 0 and intersection != self[where[0]]:
            if not any([self[where[0]].find_pt(p) for p in intersection.line_pts]):
                self.intersections.append(intersection)
            else:
                for p in intersection.line_pts:
                    pt_id = self[where[0]].add_point(p)
                for p in intersection.polygons:
                    if p not in self[where[0]].polygons:
                        self[where[0]].polygons.append(p)
                for p in intersection.plates:
                    if p not in self[where[0]].plates:
                        self[where[0]].plates.append(p)
        else:
            for p in intersection.polygons:
                if p not in self[where[0]].polygons:
                    self[where[0]].polygons.append(p)
            for p in intersection.plates:
                if p not in self[where[0]].plates:
                    self[where[0]].plates.append(p)

    def add_middle_intersection_points(self, epsilon=epsilon):
        for i in range(len(self)):
            int1 = self[i]
            for j in range(i + 1, len(self)):
                int2 = self[j]
                if int1.line.is_close(int2.line, abs_tol=epsilon):
                    continue
                int_point = intersect_2_lines(int1.line, int2.line)
                if int_point is not None:
                    if len(self[i].line_pts) != 1 and len(self[j].line_pts) == 1:
                        segment = LineSegment(self[i].line_pts[0], self[i].line_pts[-1])

                        if segment.contains_point(int_point, abs_tol=epsilon) and \
                                point_equal(int_point, self[j].line_pts[0], epsilon=epsilon):
                            pt_id = self[i].add_point(int_point)
                            pt_id = self[j].add_point(int_point)

                    elif len(self[j].line_pts) != 1 and len(self[i].line_pts) == 1:
                        segment = LineSegment(self[j].line_pts[0], self[j].line_pts[-1])
                        if segment.contains_point(int_point, abs_tol=epsilon) and \
                                point_equal(int_point, self[i].line_pts[0], epsilon=epsilon):
                            pt_id = self[i].add_point(int_point)
                            pt_id = self[j].add_point(int_point)

                    elif len(self[i].line_pts) != 1 and len(self[j].line_pts) != 1:
                        segment1 = LineSegment(self[i].line_pts[0], self[i].line_pts[-1])
                        segment2 = LineSegment(self[j].line_pts[0], self[j].line_pts[-1])
                        if segment1.contains_point(int_point, abs_tol=epsilon) and \
                                segment2.contains_point(int_point, abs_tol=epsilon):
                            pt_id = self[i].add_point(int_point)
                            pt_id = self[j].add_point(int_point)

    def find_line_from_PolyLine(self, polyline: PolygonLine, epsilon=epsilon):
        for i, intersection in enumerate(self):
            if polyline.line.is_close(intersection.line, abs_tol=epsilon) and \
                    len(intersection.line_pts) > 1:
                return i
        return None


class PointsInPolyLines:
    def __init__(self, all_points: list):
        assert type(all_points) == list, "Must be a list"
        assert all([type(p) in [np.ndarray, Point] for p in all_points]), \
            "Must be a list of arrays or points."
        self.points = []
        for p in all_points:
            i = self.find_point(p)
            if i is None:
                self.points.append(np.array(p))

    def __repr__(self):
        return f"PointsInPolyLines({len(self.points)} points)"

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def __iter__(self):
        yield from self.points

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def find_point(self, point):
        where = point_in_list(point, self.points, return_where=True)
        return None if len(where) == 0 else where[0]

    def get_pt_id(self, point):
        pid = self.find_point(point)
        return pid if pid is not None else len(self.points)

    def add_point(self, point):
        ix = self.get_pt_id(point)
        if ix == len(self.points):
            self.points.append(point)
        return ix


class PolygonLineCollection:
    def __init__(self, all_lines: list):
        assert all([type(l) == PolygonLine for l in all_lines]), \
            "Must be a list of PolygonLine objects."
        self.all_points = PointsInPolyLines([])
        self.all_lines = []
        for i, line in enumerate(all_lines):
            self.add_line(line)

    @classmethod
    def from_Polygon(cls, polygon: Polygon):
        all_lines = []
        for i, edge in enumerate(polygon.edge_lines):
            edge_points = line_in_polys(edge, polygon, polygon)
            poly_line = PolygonLine(line=edge, line_pts=edge_points,
                                    are_edges=[True],
                                    are_intersections=[False])
            all_lines.append(poly_line)
        return cls(all_lines)

    def __repr__(self):
        return f"PolygonLineCollection({len(self.all_lines)} lines)"

    def __len__(self):
        return len(self.all_lines)

    def __getitem__(self, item):
        return self.all_lines[item]

    def __iter__(self):
        yield from self.all_lines

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def line_list(self):
        return [l.line for l in self.all_lines]

    def line_id(self, line):
        where = line_in_line_list(line, self.line_list(), return_where=True)
        if len(where) == 0:
            l_id = self.__len__()
        else:
            l_id = where[0]
        return l_id

    def polyline_subset_sub_polygon(self, sub_polygon):
        node_ids, line_ids = sub_polygon
        new_poly_lines = []
        lines_in_sub_poly = [self.all_lines[i] for i in line_ids]

        for poly_line in lines_in_sub_poly:
            node_start, node_end = sorted([poly_line.point_ids().index(n)
                                           for n in node_ids if n in poly_line.point_ids()])
            line = poly_line.line
            line_pts = poly_line.point_coords()[node_start: node_end + 1]
            are_intersections = [e.is_intersection for e in poly_line.edges[node_start: node_end]]
            are_edges = [e.is_edge for e in poly_line.edges[node_start: node_end]]
            new_poly_line = PolygonLine(line, line_pts, are_edges, are_intersections)
            new_poly_lines.append(new_poly_line)
        return PolygonLineCollection(new_poly_lines)

    def all_edges(self):
        edges = []
        for l in self:
            edges += l.edges
        return edges

    def edge_pt_ids(self):
        return [sorted(edge.pt_ids()) for edge in self.all_edges()]

    def get_edge(self, pt_ids):
        for edge in self.all_edges():
            if sorted(edge.pt_ids()) == sorted(pt_ids):
                return edge
        return None

    def add_point_to_line(self, point: Node, line_id: int):
        assert type(point) == Node, "Must be a Node object."
        pt_id = self.all_points.add_point(point.point)
        point.id = pt_id
        self[line_id].add_point(point)

    def add_line(self, line: PolygonLine):
        assert type(line) == PolygonLine, "Must be a PolygonLine object."
        line = deepcopy(line)
        l_id = self.line_id(line.line)
        if l_id == self.__len__():
            line.id = l_id
            line.update_edge_ids()
            pt_ids = [self.all_points.add_point(p.point) for p in line.line_pts]
            line.change_pt_ids(pt_ids)
            self.all_lines.append(line)
        else:
            pt_ids = [self.all_points.add_point(p.point) for p in line.line_pts]
            line.change_pt_ids(pt_ids)
            self.all_lines[l_id] = line

    def add_intersection_line(self, intersection_line: PolygonLine):
        assert type(intersection_line) == PolygonLine, "Must be a PolygonLine object."
        intersection_line = deepcopy(intersection_line)
        l_id = self.line_id(intersection_line.line)
        if l_id == self.__len__():
            intersection_line.change_line_id(l_id)
            pt_ids = [self.all_points.add_point(p.point) for p in intersection_line.line_pts]
            intersection_line.change_pt_ids(pt_ids)
            self.all_lines.append(intersection_line)
            for poly_line in self:
                for point in intersection_line.line_pts:
                    if poly_line.point_in_poly_line(point.point):
                        poly_line.add_point(point)
        else:
            pt_ids = [self.all_points.add_point(p.point) for p in intersection_line.line_pts]
            intersection_line.change_pt_ids(pt_ids)
            self[l_id].add_intersection_line(intersection_line)

    def add_intersection_list(self, intersection_list: IntersectionList):
        for intersection_line in intersection_list:
            self.add_intersection_line(intersection_line)

    def get_adj_matrix(self):
        nodes = self.all_points.points
        adj_matrix = np.zeros((len(nodes), len(nodes)))
        for i, l in enumerate(self):
            for j in range(len(l) - 1):
                adj_matrix[l[j].id, l[j + 1].id] = 1
                adj_matrix[l[j + 1].id, l[j].id] = 1
        return adj_matrix

    def get_graph(self):
        G = nx.Graph(self.get_adj_matrix())
        for i, node in enumerate(self.all_points.points):
            G.nodes[i]['point'] = node
        return G
