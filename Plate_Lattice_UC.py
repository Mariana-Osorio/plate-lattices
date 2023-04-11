from skspatial.objects import Plane, Line, LineSegment, Point, Points, Vector
from skspatial.typing import array_like
import numpy as np
from itertools import product
from plate_lattices import *


class Polygon:
    def __init__(self):
        self.edge_lines = []
        self.vertices = None


class Plate:
    def __init__(self, normal: array_like, plane: Plane, polygons):
        self.normal = normal
        self.plane = plane
        self.polygons = []
        self.type = None


class PlateLatticeUC:
    def __init__(self):
        self.polygons = []
        self.planes = []
        self.plane_id = []
        self.normals = []
        self.nn_polygons = None
        self.nn_planes = None
        self.nn_plane_id = None
        self.nn_normals = None
        self.uc_lines = None

    def __len__(self):
        return len(self.normals)

    def has_normal(self, normal, epsilon=1e-10):
        dup_normals = [1 for n in self.normals if Vector(n).is_parallel(normal, abs_tol=epsilon)]
        return True if len(dup_normals) != 0 else False

    def has_plane(self, plane, epsilon=1e-10):
        if self.nn_planes is None:
            self.nn_uc()
        dup_planes = [1 for p in self.nn_planes if p.is_close(plane, abs_tol=epsilon)]
        return True if len(dup_planes) != 0 else False

    def add_to_plate_lattice_uc(self, normal, polygons):
        self.plane_id += len(polygons) * [len(self.normals)]
        self.normals.append(normal)
        self.polygons += polygons
        self.planes += [get_plane_from_poly(p, normal=normal) for p in polygons]

        if self.nn_polygons is not None:
            self.nn_plane_id += len(polygons) * [len(self.nn_normals)]
            self.nn_normals.append(normal)
            self.nn_polygons += polygons
            self.nn_planes += [get_plane_from_poly(p, normal=normal) for p in polygons]

    def add_poly_to_plate_lattice_uc_nn(self, plane_id, polygon):
        if self.nn_polygons is None:
            raise Exception("nn_uc not initialized.")
        self.nn_polygons.append(polygon)
        self.nn_planes.append(get_plane_from_poly(polygon, normal=self.normals[plane_id]))
        self.nn_plane_id.append(plane_id)

    def get_neighbor(self, idx):
        assert type(idx) == tuple and len(idx) == 3 and type(idx[0]) == int \
               and type(idx[1]) == int and type(idx[2]) == int, "Not valid idx"
        polygons = [p + idx for p in self.polygons]
        planes = [get_plane_from_poly(polygons[i], normal=self.normals[plane_id])
                  for i, plane_id in enumerate(self.plane_id)]
        return polygons, planes

    def nn_uc(self):
        self.nn_polygons = []
        self.nn_planes = []
        self.nn_plane_id = []
        self.nn_normals = self.normals.copy()

        for neighbor in product([0, 1, -1], [0, 1, -1], [0, 1, -1]):
            polygons, planes = self.get_neighbor(neighbor)
            for i, plane_id in enumerate(self.plane_id):
                if point_greater(polygons[i], 0.0, inclusive=True) and point_smaller(polygons[i], 1.0, inclusive=True):
                    self.add_poly_to_plate_lattice_uc_nn(plane_id, polygons[i])

    def get_uc_lines(self, polygons):
        if self.nn_polygons is None:
            self.nn_uc()

        lines = {'yz': [], 'xz': [], 'xy': []}
        for vertices in polygons:
            for val in [0., 1.]:
                for plane in range(3):
                    if len(vertices[vertices[:, plane] == val]) >= 2:
                        line = vertices[vertices[:, plane] == val][:, [i for i in [0, 1, 2] if i != plane]]
                        if not array_in_list(line, lines[list(lines.keys())[plane]]):
                            lines[list(lines.keys())[plane]] += [line]

        return lines


class PlateIntersections:
    def __init__(self, plate_lattice_uc: PlateLatticeUC):
        self.plate_lattice_uc = plate_lattice_uc
        self.intersection_lines = []
        self.intersection_line_pts = []
        self.plane_id = []
        self.polygon_id = []
        self.polygon = None
        self.polygon_plane = None

    def __repr__(self):
        if self.polygon is None:
            return f"PlateIntersections({len(self.intersection_lines)} int lines, plate lattice)"
        else:
            return f"PlateIntersections({len(self.intersection_lines)} int lines, polygon with plate lattice)"

    def __len__(self):
        return len(self.intersection_lines)

    @classmethod
    def plate_lattice_uc_intersections(cls, plate_lattice_uc, epsilon=1e-10):
        pi = cls(plate_lattice_uc)
        for i, plane1 in enumerate(pi.plate_lattice_uc.planes):
            for j, plane2 in list(enumerate(pi.plate_lattice_uc.planes))[i+1:]:
                if plane1.is_close(plane2, abs_tol=epsilon):
                    continue
                line_intersection = intersect_2_planes(plane1, plane2)
                if line_intersection is not None:
                    line_points = line_in_polygons(line_intersection,
                                                   pi.plate_lattice_uc.polygons[i],
                                                   pi.plate_lattice_uc.polygons[j])
                    k = line_in_line_list(line_intersection, pi.intersection_lines, return_where=True)
                    if len(k) == 0 and len(line_points) != 0:
                        pi.intersection_lines.append(line_intersection)
                        pi.intersection_line_pts.append(line_points)
                        pi.plane_id.append([pi.plate_lattice_uc.plane_id[i],
                                            pi.plate_lattice_uc.plane_id[j]])
                        pi.polygon_id.append([i, j])

                    elif len(k) != 0 and len(line_points) != 0:
                        if pi.plate_lattice_uc.plane_id[j] not in pi.plane_id[k[0]]:
                            pi.plane_id[k[0]] += [pi.plate_lattice_uc.plane_id[j]]
                        if j not in pi.polygon_id[k[0]]:
                            pi.polygon_id[k[0]] += [j]
                        if not arrays_equal(pi.intersection_line_pts[k[0]], line_points):
                            pi.intersection_line_pts[k[0]] = np.vstack((pi.intersection_line_pts[k[0]],
                                                                        line_points))
        return pi

    @classmethod
    def plate_lattice_uc_intersections_nn(cls, plate_lattice_uc, epsilon=1e-10):
        pi = cls(plate_lattice_uc)
        pi.plate_lattice_uc.nn_uc()
        for i, plane1 in enumerate(pi.plate_lattice_uc.nn_planes):
            for j, plane2 in list(enumerate(pi.plate_lattice_uc.nn_planes))[i+1:]:
                if plane1.is_close(plane2, abs_tol=epsilon):
                    continue
                line_intersection = intersect_2_planes(plane1, plane2)
                if line_intersection is not None:
                    line_points = line_in_polygons(line_intersection,
                                                   pi.plate_lattice_uc.nn_polygons[i],
                                                   pi.plate_lattice_uc.nn_polygons[j])
                    k = line_in_line_list(line_intersection, pi.intersection_lines, return_where=True)
                    if len(k) == 0 and len(line_points) != 0:
                        pi.intersection_lines.append(line_intersection)
                        pi.intersection_line_pts.append(line_points)
                        pi.plane_id.append([pi.plate_lattice_uc.nn_plane_id[i],
                                            pi.plate_lattice_uc.nn_plane_id[j]])
                        pi.polygon_id.append([i, j])

                    elif len(k) != 0 and len(line_points) != 0:
                        if pi.plate_lattice_uc.nn_plane_id[j] not in pi.plane_id[k[0]]:
                            pi.plane_id[k[0]] += [pi.plate_lattice_uc.nn_plane_id[j]]
                        if j not in pi.polygon_id[k[0]]:
                            pi.polygon_id[k[0]] += [j]
                        if not arrays_equal(pi.intersection_line_pts[k[0]], line_points):
                            pi.intersection_line_pts[k[0]] = np.vstack((pi.intersection_line_pts[k[0]],
                                                                        line_points))
        return pi

    @classmethod
    def intersect_w_polygon(cls, plate_lattice_uc, polygon: array_like, polygon_plane: Plane, epsilon=1e-10):
        pi = cls(plate_lattice_uc)
        pi.polygon = polygon
        pi.polygon_plane = polygon_plane
        pi.plate_lattice_uc.nn_uc()
        polygon_id = len(pi.plate_lattice_uc.nn_polygons)
        plane_id = len(np.unique(pi.plate_lattice_uc.nn_plane_id))
        for i, plane in enumerate(pi.plate_lattice_uc.nn_planes):
            if polygon_plane.is_close(plane, abs_tol=epsilon):
                raise Exception("Plate is already part of plate lattice.")
            line_intersection = intersect_2_planes(polygon_plane, plane)
            if line_intersection is not None:
                line_points = line_in_polygons(line_intersection, pi.polygon, pi.plate_lattice_uc.nn_polygons[i])
                k = line_in_line_list(line_intersection, pi.intersection_lines, return_where=True)
                if len(k) == 0 and len(line_points) != 0:
                    pi.intersection_lines.append(line_intersection)
                    pi.intersection_line_pts.append(line_points)
                    pi.plane_id.append([pi.plate_lattice_uc.nn_plane_id[i], plane_id])
                    pi.polygon_id.append([i, polygon_id])

                elif len(k) != 0 and len(line_points) != 0:
                    if pi.plate_lattice_uc.nn_plane_id[i] not in pi.plane_id[k[0]]:
                        pi.plane_id[k[0]] += [pi.plate_lattice_uc.nn_plane_id[i]]
                    if i not in pi.polygon_id[k[0]]:
                        pi.polygon_id[k[0]] += [i]
                    if not arrays_equal(pi.intersection_line_pts[k[0]], line_points):
                        pi.intersection_line_pts[k[0]] = np.vstack((pi.intersection_line_pts[k[0]],
                                                                    line_points))
        return pi

    def add_intersection_points(self, epsilon=1e-10):
        for i in range(len(self)):
            l1 = self.intersection_lines[i]
            for j in range(i+1, len(self)):
                l2 = self.intersection_lines[j]
                if l1.is_close(l2, abs_tol=epsilon):
                    continue
                int_point = intersect_2_lines(l1, l2)
                if int_point is not None:
                    if len(self.intersection_line_pts[i]) == 1:
                        continue
                    segment = LineSegment(self.intersection_line_pts[i][0],
                                          self.intersection_line_pts[i][1])
                    if segment.contains_point(int_point, abs_tol=epsilon):
                        if not point_in_array(int_point, self.intersection_line_pts[i]):
                            self.intersection_line_pts[i] = np.vstack((self.intersection_line_pts[i], int_point))
                        if not point_in_array(int_point, self.intersection_line_pts[j]):
                            self.intersection_line_pts[j] = np.vstack((self.intersection_line_pts[j], int_point))


class PointInLine:
    def __init__(self, point, line_ix):
        self.point = point
        self.line_ix = line_ix

    def __repr__(self):
        return f"PointInLine(point={np.around(self.point,3)}, line_ix={self.line_ix})"


class PointsInLine:
    def __init__(self):
        self.points = []
        self.pt_ids = []
        self.line_ixs = []

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return f"PointsInLine({self.__len__()} points)"

    def get_point_id(self, point: PointInLine):
        where = point_in_list(point.point, self.points, return_where=True)
        if len(where) == 0:
            return self.__len__()
        else:
            return self.pt_ids[where[0]]

    def add_point(self, point: PointInLine):
        pt_id = self.get_point_id(point)
        where = point_in_list(point.point, self.points, return_where=True)
        if len(where) == 0:
            self.points.append(point.point)
            self.pt_ids.append(pt_id)
            self.line_ixs.append([point.line_ix])
        else:
            if point.line_ix not in self.line_ixs:
                self.line_ixs[where[0]].append(point.line_ix)
        return pt_id

    def print_points(self):
        for i in range(self.__len__()):
            print(f"Point {i}: {np.around(self.points[i],3)}")

    def to_dict(self):
        return {i: self.points[i] for i in range(len(self.points))}


class PolygonLine:
    def __init__(self, line: Line, points: array_like, is_edge: bool, is_intersection: bool, p_id_start: int = 0):
        self.line = line
        self.points = points
        self.point_ids = list(range(p_id_start, p_id_start + len(points)))
        self.edge = is_edge
        self.intersection = is_intersection

    def __repr__(self):
        return f"PolygonLine(Line(point={np.around(self.line.point, 3)}, " \
               f"direction={np.around(self.line.direction, 3)}))"

    def change_point_id(self, point_ix, new_id):
        self.point_ids[point_ix] = new_id

    def change_point_ids(self, new_ids):
        if len(new_ids) != len(self.point_ids):
            raise Exception("New ids must have the same length as the current ids.")
        self.point_ids = new_ids

    def add_point(self, point, point_id):
        if not point_in_array(point, self.points):
            self.points = np.vstack((self.points, point))
            self.point_ids.append(point_id)


class PolygonIntersections:
    def __init__(self, polygon: array_like, polygon_plane: Plane, polygon_edges):
        self.polygon = polygon
        self.polygon_plane = polygon_plane
        self.polygon_edges = polygon_edges
        self.lines = []
        self.all_points = PointsInLine()
        self.edges_to_lines()

    def __repr__(self):
        return f"PolygonIntersections(Plane(point={np.around(self.polygon_plane.point, 3)}, " \
               f"normal={np.around(self.polygon_plane.normal, 3)}))"

    def __len__(self):
        return len(self.lines)

    def edges_to_lines(self):
        for i, edge in enumerate(self.polygon_edges):
            edge_points = line_in_polygons(edge, self.polygon, self.polygon)
            poly_line = PolygonLine(edge, edge_points, True, False, p_id_start=i * 2)
            pt1 = PointInLine(poly_line.points[0], len(self.lines))
            pt2 = PointInLine(poly_line.points[1], len(self.lines))

            pt_id1 = self.all_points.add_point(pt1)
            pt_id2 = self.all_points.add_point(pt2)
            poly_line.change_point_ids([pt_id1, pt_id2])
            self.lines.append(poly_line)

    def all_lines(self):
        return [line.line for line in self.lines]

    def line_pt_ids(self):
        return [line.point_ids for line in self.lines]

    def add_point_to_line(self, point: PointInLine):
        line_ix = point.line_ix
        point_id = self.all_points.add_point(point)
        self.lines[line_ix].add_point(point.point, point_id)

    def add_line(self, line: PolygonLine):
        if line.line not in self.all_lines():
            line_id = len(self.lines)
            for ix, point in enumerate(line.points):
                pt_id = line.point_ids[ix]
                point = PointInLine(point, line_id)
                point_id = self.all_points.add_point(point)
                if pt_id != point_id:
                    line.change_point_id(ix, point_id)
            self.lines.append(line)
            return line

    def add_intersection_lines(self, intersections: PlateIntersections, epsilon = 1e-10):
        for i, int_line in enumerate(intersections.intersection_lines):
            poly_line = PolygonLine(int_line,
                                    intersections.intersection_line_pts[i],
                                    is_edge=False,
                                    is_intersection=True)
            where = line_in_line_list(poly_line.line,
                                      self.all_lines(),
                                      return_where=True)
            if len(where) != 0:
                self.lines[where[0]].intersection = True
                for pt in intersections.intersection_line_pts[i]:
                    pt = PointInLine(pt, where[0])
                    self.add_point_to_line(pt)
            else:
                poly_line = self.add_line(poly_line)
                for j, edge in enumerate(self.polygon_edges):
                    edge_points = line_pts_in_uc(edge)
                    edge_segment = LineSegment(edge_points[0], edge_points[1])
                    for pt, pt_id in zip(poly_line.points, poly_line.point_ids):
                        if edge_segment.contains_point(pt, abs_tol=epsilon):
                            pt = PointInLine(pt, j)
                            self.add_point_to_line(pt)

    def order_lines(self):
        for i in range(len(self)):
            dir_vec = self.lines[i].line.direction
            ix_order = arg_order_points_on_line(self.lines[i].points, dir_vec)
            self.lines[i].points = self.lines[i].points[ix_order]
            self.lines[i].point_ids = [self.lines[i].point_ids[k] for k in ix_order]

    def get_adj_matrix(self):
        nodes = self.all_points.to_dict()
        adj_matrix = np.zeros((len(nodes), len(nodes)))
        for i, l in enumerate(self.lines):
            for j in range(len(l.points) - 1):
                adj_matrix[l.point_ids[j], l.point_ids[j + 1]] = 1
                adj_matrix[l.point_ids[j + 1], l.point_ids[j]] = 1
        return adj_matrix
