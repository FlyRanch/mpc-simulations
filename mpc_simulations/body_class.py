from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl import NURBS
from geomdl import tessellate
import numpy as np
import numpy.matlib
import os
import copy
import time
import json
import h5py

class NURBS_object():

    def __init__(self, obj_name, ctrl_pts, knt_vectors, grid_size, surf_degree, N_pts_u, N_pts_v):
        self.object_name = obj_name
        self.control_pts = ctrl_pts
        self.grid_size = grid_size
        self.knot_vector_u = knt_vectors[0]
        self.knot_vector_v = knt_vectors[1]
        self.degree_u = surf_degree[0]
        self.degree_v = surf_degree[1]
        self.N_u = N_pts_u
        self.N_v = N_pts_v
        self.surf = NURBS.Surface()
        self.surf.tessellator = tessellate.TriangularTessellate()
        self.update_NURBS_surf()

    def set_N_u(self,N_u):
        self.N_u = N_u

    def set_N_v(self,N_v):
        self.N_v = N_v

    def update_NURBS_surf(self):
        self.surf.reset()
        self.surf.degree_u = self.degree_u
        self.surf.degree_v = self.degree_v
        control_pts_now = []
        for ctrl_pt in self.control_pts:
            control_pts_now.append(ctrl_pt)
        self.surf.set_ctrlpts(self.control_pts,self.grid_size[0],self.grid_size[1])
        self.surf.knotvector_u = self.knot_vector_u
        self.surf.knotvector_v = self.knot_vector_v
        self.surf.sample_size_u = self.N_u
        self.surf.sample_size_v = self.N_v

    def return_polydata(self):
        self.surf.tessellate()
        triangles = self.surf.tessellator.faces
        vertices = self.surf.tessellator.vertices
        #print('triangles')
        #triangles = self.surf.tessellator.triangles
        #print('success')
        poly_points = vtk.vtkPoints()
        poly_triangles = vtk.vtkCellArray()
        count = 0
        for t in triangles:
            for v in t.vertices:
                poly_points.InsertNextPoint(v.x,v.y,v.z)
            tri = vtk.vtkTriangle()
            v_ids = t.vertex_ids
            tri.GetPointIds().SetId(0, count)
            tri.GetPointIds().SetId(1, count+1)
            tri.GetPointIds().SetId(2, count+2)
            poly_triangles.InsertNextCell(tri)
            count += 3
        polydat = vtk.vtkPolyData()
        polydat.SetPoints(poly_points)
        polydat.SetPolys(poly_triangles)
        poly_object = vtk.vtkCleanPolyData()
        poly_object.SetInputData(polydat)
        poly_object.Update()
        return poly_object.GetOutput()

# Construct a model of the thorax:
class BodyModel():

    def __init__(self):
        self.body_clr = (0.01,0.01,0.01)
        self.head()
        self.thorax()
        self.abdomen()

    def head(self):
        # construct head:
        comp_name = 'head'
        ctrl_pts = [[0.6, 0.0, 0.0, 1.0], [0.3818, 0.0, -0.1697, 0.6364], [0.27, 0.0, -0.45, 1.0], [0.1061, 0.0, -0.2970, 0.7071], 
            [0.0, 0.0, 0.0, 1.0], [0.4243, 0.0, 0.0, 0.7071], [0.27, 0.2, -0.12, 0.5], [0.1909, 0.2828, -0.3182, 0.7071], [0.075, 0.2, -0.21, 0.5], 
            [0.0, 0.0, 0.0, 0.7071], [0.6, 0.0, 0.0, 1.0], [0.3818, 0.2828, 0.0, 0.7071], [0.24, 0.4, 0.15, 1.0], [0.0636, 0.2828, 0.1061, 0.6364], 
            [0.0, 0.0, 0.0, 1.0], [0.4243, 0.0, 0.0, 0.7071], [0.27, 0.2, 0.12, 0.5], [0.1697, 0.2828, 0.2758, 0.7071], [0.045, 0.2, 0.18, 0.4], 
            [0.0, 0.0, 0.0, 0.7071], [0.6, 0.0, 0.0, 1.0], [0.3818, 0.0, 0.1697, 0.7071], [0.24, 0.0, 0.39, 1.0], [0.0636, 0.0, 0.2546, 0.5657], 
            [0.0, 0.0, 0.0, 1.0], [0.4243, 0.0, 0.0, 0.7071], [0.27, -0.2, 0.12, 0.5], [0.1697, -0.2828, 0.2758, 0.7071], [0.045, -0.2, 0.18, 0.4], 
            [0.0, 0.0, 0.0, 0.7071], [0.6, 0.0, 0.0, 1.0], [0.3818, -0.2828, 0.0, 0.7071], [0.24, -0.4, 0.15, 1.0], [0.0636, -0.2828, 0.1061, 0.6364], 
            [0.0, 0.0, 0.0, 1.0], [0.4243, 0.0, 0.0, 0.7071], [0.27, -0.2, -0.12, 0.5], [0.1909, -0.2828, -0.3182, 0.7071], [0.075, -0.2, -0.21, 0.5], 
            [0.0, 0.0, 0.0, 0.7071], [0.6, 0.0, 0.0, 1.0], [0.3818, 0.0, -0.1697, 0.6364], [0.27, 0.0, -0.45, 1.0], [0.1061, 0.0, -0.2970, 0.7071], 
            [0.0, 0.0, 0.0, 1.0]]
        knt_vectors = [[0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]]
        grid_size = [9,5]
        surf_degree = [2,2]
        N_pts_u = 20
        N_pts_v = 20
        self.head_Surf = NURBS_object(comp_name,ctrl_pts,knt_vectors,grid_size,surf_degree,N_pts_u,N_pts_v)
        mesh = self.head_Surf.return_polydata()

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(mesh)

        self.head_Actor = vtk.vtkActor()
        self.head_Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
        #self.head_Actor.GetProperty().SetRepresentationToWireframe()
        self.head_Actor.SetMapper(Mapper)

    def write_stl_head(self,save_dir,scale_in):
        transform = vtk.vtkTransform()
        transform.Scale(scale_in,scale_in,scale_in)
        stlWriter = vtk.vtkSTLWriter()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        mesh = self.head_Surf.return_polydata()
        transformPD.SetInputData(mesh)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'head.stl'))
        stlWriter.Write()

    def thorax(self):
        # construct thorax
        comp_name = 'thorax'
        ctrl_pts = [[0.86, 0.0, 0.0, 1.0], [0.4243, 0.0, -0.3818, 0.4950], 
            [-0.12, 0.0, -0.9, 1.0], [-0.1909, 0.0, -0.5091, 0.4667], [-0.24, 0.0, 0.0, 1.0], [0.6081, 0.0, 0.0, 0.7071], [0.3, 0.3, -0.27, 0.5], 
            [-0.0849, 0.2121, -0.6364, 0.7071], [-0.135, 0.15, -0.36, 0.5], [-0.1697, 0.0, 0.0, 0.7071], [0.86, 0.0, 0.0, 1.0], [0.6081, 0.4243, 0.0, 0.6010], 
            [0.0, 0.6, 0.0, 1.0], [-0.1697, 0.4243, 0.0, 0.6010], [-0.24, 0.0, 0.0, 1.0], [0.6081, 0.0, 0.0, 0.7071], [0.39, 0.3, 0.21, 0.5], 
            [0.2546, 0.4243, 0.4243, 0.6364], [-0.135, 0.3, 0.075, 0.2], [-0.1697, 0.0, 0.0, 0.7071], [0.86, 0.0, 0.0, 1.0], [0.5515, 0.0, 0.2970, 0.5657], 
            [0.36, 0.0, 0.6, 1.1], [-0.1909, 0.0, 0.1061, 0.2828], [-0.24, 0.0, 0.0, 1.0], [0.6081, 0.0, 0.0, 0.7071], [0.39, -0.3, 0.21, 0.5], 
            [0.2546, -0.4243, 0.4243, 0.6364], [-0.135, -0.3, 0.075, 0.2], [-0.1697, 0.0, 0.0, 0.7071], [0.86, 0.0, 0.0, 1.0], [0.6081, -0.4243, 0.0, 0.6010], 
            [0.0, -0.6, 0.0, 1.0], [-0.1697, -0.4243, 0.0, 0.6010], [-0.24, 0.0, 0.0, 1.0], [0.6081, 0.0, 0.0, 0.7071], [0.3, -0.3, -0.27, 0.5], 
            [-0.0849, -0.2121, -0.6364, 0.7071], [-0.135, -0.15, -0.36, 0.5], [-0.1697, 0.0, 0.0, 0.7071], [0.86, 0.0, 0.0, 1.0], [0.4243, 0.0, -0.3818, 0.4950], 
            [-0.12, 0.0, -0.9, 1.0], [-0.1909, 0.0, -0.5091, 0.4667], [-0.24, 0.0, 0.0, 1.0]]
        knt_vectors = [[0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]]
        grid_size = [9,5]
        surf_degree = [2,2]
        N_pts_u = 20
        N_pts_v = 20
        self.thorax_Surf = NURBS_object(comp_name,ctrl_pts, knt_vectors,grid_size,surf_degree,N_pts_u,N_pts_v)
        mesh = self.thorax_Surf.return_polydata()

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(mesh)

        self.thorax_Actor = vtk.vtkActor()
        self.thorax_Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
        #self.thorax_Actor.GetProperty().SetRepresentationToWireframe()
        self.thorax_Actor.SetMapper(Mapper)

    def write_stl_thorax(self,save_dir,scale_in):
        transform = vtk.vtkTransform()
        transform.Scale(scale_in,scale_in,scale_in)
        stlWriter = vtk.vtkSTLWriter()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        mesh = self.thorax_Surf.return_polydata()
        transformPD.SetInputData(mesh)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'thorax.stl'))
        stlWriter.Write()

    def abdomen(self):
        # construct abdomen
        comp_name = 'abdomen'
        ctrl_pts = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.3182, 0.5657], [-0.72, 0.0, -0.45, 1.0], 
            [-1.1667, 0.0, -0.3182, 0.7071], [-1.65, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.7071], [0.0, 0.25, -0.225, 0.4], [-0.5091, 0.3536, -0.3182, 0.7071], [-0.825, 0.25, -0.225, 0.5], 
            [-1.1667, 0.0, 0.0, 0.7071], [0.0, 0.0, 0.0, 1.0], [0.0, 0.3536, 0.0, 0.5657], [-0.72, 0.5, 0.0, 1.0], 
            [-1.1667, 0.3536, 0.0, 0.7071], [-1.65, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.7071], [0.0, 0.25, 0.27, 0.4], 
            [-0.5091, 0.3536, 0.3818, 0.7071], [-0.825, 0.25, 0.27, 0.5], [-1.1667, 0.0, 0.0, 0.7071], 
            [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.3818, 0.5657], [-0.72, 0.0, 0.54, 1.0], [-1.1667, 0.0, 0.3818, 0.7071], 
            [-1.65, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.7071], [0.0, -0.25, 0.27, 0.4], [-0.5091, -0.3536, 0.3818, 0.7071], 
            [-0.825, -0.25, 0.27, 0.5], [-1.1667, 0.0, 0.0, 0.7071], [0.0, 0.0, 0.0, 1.0], [0.0, -0.3536, 0.0, 0.5657], 
            [-0.72, -0.5, 0.0, 1.0], [-1.1667, -0.3536, 0.0, 0.7071], [-1.65, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.7071], 
            [0.0, -0.25, -0.225, 0.4], [-0.5091, -0.3536, -0.3182, 0.7071], [-0.825, -0.25, -0.225, 0.5], 
            [-1.1667, 0.0, 0.0, 0.7071], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -0.3182, 0.5657], [-0.72, 0.0, -0.45, 1.0], 
            [-1.1667, 0.0, -0.3182, 0.7071], [-1.65, 0.0, 0.0, 1.0]]
        knt_vectors = [[0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0]]
        grid_size = [9,5]
        surf_degree = [2,2]
        N_pts_u = 20
        N_pts_v = 20
        self.abdomen_Surf = NURBS_object(comp_name,ctrl_pts, knt_vectors,grid_size,surf_degree,N_pts_u,N_pts_v)
        mesh = self.abdomen_Surf.return_polydata()

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(mesh)

        self.abdomen_Actor = vtk.vtkActor()
        self.abdomen_Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
        #self.abdomen_Actor.GetProperty().SetRepresentationToWireframe()
        self.abdomen_Actor.SetMapper(Mapper)

    def write_stl_abdomen(self,save_dir,scale_in):
        transform = vtk.vtkTransform()
        transform.Scale(scale_in,scale_in,scale_in)
        stlWriter = vtk.vtkSTLWriter()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        mesh = self.abdomen_Surf.return_polydata()
        transformPD.SetInputData(mesh)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir,'abdomen.stl'))
        stlWriter.Write()

    def add_actors(self,renderer):
        renderer.AddActor(self.head_Actor)
        renderer.AddActor(self.thorax_Actor)
        renderer.AddActor(self.abdomen_Actor)

    def transform_head(self,s_in):
        q_norm = np.sqrt(pow(s_in[0],2)+pow(s_in[1],2)+pow(s_in[2],2)+pow(s_in[3],2))
        q0 = s_in[0]/q_norm
        q1 = s_in[1]/q_norm
        q2 = s_in[2]/q_norm
        q3 = s_in[3]/q_norm
        tx = s_in[4]
        ty = s_in[5]
        tz = s_in[6]
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        M_vtk = vtk.vtkMatrix4x4()
        M_vtk.SetElement(0,0,M[0,0])
        M_vtk.SetElement(0,1,M[0,1])
        M_vtk.SetElement(0,2,M[0,2])
        M_vtk.SetElement(0,3,M[0,3])
        M_vtk.SetElement(1,0,M[1,0])
        M_vtk.SetElement(1,1,M[1,1])
        M_vtk.SetElement(1,2,M[1,2])
        M_vtk.SetElement(1,3,M[1,3])
        M_vtk.SetElement(2,0,M[2,0])
        M_vtk.SetElement(2,1,M[2,1])
        M_vtk.SetElement(2,2,M[2,2])
        M_vtk.SetElement(2,3,M[2,3])
        M_vtk.SetElement(3,0,M[3,0])
        M_vtk.SetElement(3,1,M[3,1])
        M_vtk.SetElement(3,2,M[3,2])
        M_vtk.SetElement(3,3,M[3,3])
        self.head_Actor.SetUserMatrix(M_vtk)
        self.head_Actor.Modified()
        return M

    def transform_thorax(self,s_in):
        q_norm = np.sqrt(pow(s_in[0],2)+pow(s_in[1],2)+pow(s_in[2],2)+pow(s_in[3],2))
        q0 = s_in[0]/q_norm
        q1 = s_in[1]/q_norm
        q2 = s_in[2]/q_norm
        q3 = s_in[3]/q_norm
        tx = s_in[4]
        ty = s_in[5]
        tz = s_in[6]
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        M_vtk = vtk.vtkMatrix4x4()
        M_vtk.SetElement(0,0,M[0,0])
        M_vtk.SetElement(0,1,M[0,1])
        M_vtk.SetElement(0,2,M[0,2])
        M_vtk.SetElement(0,3,M[0,3])
        M_vtk.SetElement(1,0,M[1,0])
        M_vtk.SetElement(1,1,M[1,1])
        M_vtk.SetElement(1,2,M[1,2])
        M_vtk.SetElement(1,3,M[1,3])
        M_vtk.SetElement(2,0,M[2,0])
        M_vtk.SetElement(2,1,M[2,1])
        M_vtk.SetElement(2,2,M[2,2])
        M_vtk.SetElement(2,3,M[2,3])
        M_vtk.SetElement(3,0,M[3,0])
        M_vtk.SetElement(3,1,M[3,1])
        M_vtk.SetElement(3,2,M[3,2])
        M_vtk.SetElement(3,3,M[3,3])
        self.thorax_Actor.SetUserMatrix(M_vtk)
        self.thorax_Actor.Modified()
        return M

    def transform_abdomen(self,s_in):
        q_norm = np.sqrt(pow(s_in[0],2)+pow(s_in[1],2)+pow(s_in[2],2)+pow(s_in[3],2))
        q0 = s_in[0]/q_norm
        q1 = s_in[1]/q_norm
        q2 = s_in[2]/q_norm
        q3 = s_in[3]/q_norm
        tx = s_in[4]
        ty = s_in[5]
        tz = s_in[6]
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        M_vtk = vtk.vtkMatrix4x4()
        M_vtk.SetElement(0,0,M[0,0])
        M_vtk.SetElement(0,1,M[0,1])
        M_vtk.SetElement(0,2,M[0,2])
        M_vtk.SetElement(0,3,M[0,3])
        M_vtk.SetElement(1,0,M[1,0])
        M_vtk.SetElement(1,1,M[1,1])
        M_vtk.SetElement(1,2,M[1,2])
        M_vtk.SetElement(1,3,M[1,3])
        M_vtk.SetElement(2,0,M[2,0])
        M_vtk.SetElement(2,1,M[2,1])
        M_vtk.SetElement(2,2,M[2,2])
        M_vtk.SetElement(2,3,M[2,3])
        M_vtk.SetElement(3,0,M[3,0])
        M_vtk.SetElement(3,1,M[3,1])
        M_vtk.SetElement(3,2,M[3,2])
        M_vtk.SetElement(3,3,M[3,3])
        self.abdomen_Actor.SetUserMatrix(M_vtk)
        self.abdomen_Actor.Modified()
        return M

    def scale_head(self,scale_in):
        self.head_Actor.SetScale(scale_in,scale_in,scale_in)
        self.head_Actor.Modified()

    def scale_thorax(self,scale_in):
        self.thorax_Actor.SetScale(scale_in,scale_in,scale_in)
        self.thorax_Actor.Modified()

    def scale_abdomen(self,scale_in):
        self.abdomen_Actor.SetScale(scale_in,scale_in,scale_in)
        self.abdomen_Actor.Modified()

    def set_Color(self,color_vec):
        self.body_clr = color_vec[0]
        self.head_Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
        self.head_Actor.Modified()
        self.thorax_Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
        self.thorax_Actor.Modified()
        self.abdomen_Actor.GetProperty().SetColor(self.body_clr[0],self.body_clr[1],self.body_clr[2])
        self.abdomen_Actor.Modified()
