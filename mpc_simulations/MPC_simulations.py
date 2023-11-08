import sys
import os
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import time
import math
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, GaussianNoise, Flatten, Conv2D
from keras.models import Model
from keras.losses import mse
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.utils.data_utils import Sequence
from keras import optimizers
from keras import backend as K
from scipy import linalg
import scipy.special
from scipy import interpolate

# Import do_mpc package:
from casadi import *
import do_mpc

# Local package imports
from .mpl_functions import adjust_spines
from .body_class import BodyModel
from .wing_twist_class import WingModel_L
from .wing_twist_class import WingModel_R

# WBD can get rid of this
# ---------------------------------------------
#cwd = os.getcwd()
#print(cwd)
#os.chdir(cwd+'/drosophila_model')
#from body_class import BodyModel
#from wing_twist_class import WingModel_L
#from wing_twist_class import WingModel_R
# ---------------------------------------------

class MPC_simulations():

    def __init__(self):
        self.N_pol_theta = 20
        self.N_pol_eta = 24
        self.N_pol_phi = 16
        self.N_pol_xi = 20
        self.N_const = 3
        self.grey = (0.5,0.5,0.5)
        self.black = (0.1,0.1,0.1)
        self.red = (1.0,0.0,0.0)
        self.blue = (0.0,0.0,1.0)
        self.dx = 0.01
        self.body_density = 1.2e-6
        self.wing_density = self.body_density*5.4e-4*(1.0/self.dx)
        self.g = 9800.0 # mm/s^2
        self.rho_air = 1.18e-9 #kg/mm^3
        self.m_names = ['b1','b2','b3','i1','i2','iii1','iii24','iii3','hg1','hg2','hg3','hg4','freq']
        self.c_muscle = ['darkred','red','orangered','dodgerblue','blue','darkgreen','lime','springgreen','indigo','fuchsia','mediumorchid','deeppink']
        # current working directory
        self.working_dir = os.getcwd() 
        self.plots_dir = os.path.join(self.working_dir, 'plots')
        self.weight_file = os.path.join(self.working_dir, 'muscle_wing_weights_new.h5')

    def Renderer(self):
        self.ren = vtk.vtkRenderer()
        self.ren.SetUseDepthPeeling(True)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

    def ConstructModel(self):
        # Load body and wing models:
        self.body_mdl = BodyModel()
        self.body_mdl.add_actors(self.ren)
        self.wing_mdl_L = WingModel_L()
        self.wing_mdl_L.add_actors(self.ren)
        self.wing_mdl_R = WingModel_R()
        self.wing_mdl_R.add_actors(self.ren)
        time.sleep(0.001)

    def ScaleModel(self,scale_in):
        self.scale = scale_in
        self.body_mdl.scale_head(scale_in[0])
        self.body_mdl.scale_thorax(scale_in[1])
        self.body_mdl.scale_abdomen(scale_in[2])
        self.wing_mdl_L.scale_wing(scale_in[3])
        self.wing_mdl_R.scale_wing(scale_in[4])

    def SetModelState(self,s_head_in,s_thorax_in,s_abdomen_in,s_wing_L_in,s_wing_R_in):
        # head
        self.body_mdl.transform_head(s_head_in)
        self.s_head = np.copy(s_head_in)
        # thorax
        self.body_mdl.transform_thorax(s_thorax_in)
        self.s_thorax = np.copy(s_thorax_in)
        # abdomen
        self.body_mdl.transform_abdomen(s_abdomen_in)
        self.s_abdomen = np.copy(s_abdomen_in)
        # wing L
        self.wing_mdl_L.transform_wing(s_wing_L_in)
        self.s_wing_L = np.copy(s_wing_L_in)
        # wing R
        self.wing_mdl_R.transform_wing(s_wing_R_in)
        self.s_wing_R = np.copy(s_wing_R_in)

    def ComputeInertia(self):
        # get polydata
        self.head_poly       = self.body_mdl.head_Surf.return_polydata()
        self.thorax_poly  = self.body_mdl.thorax_Surf.return_polydata()
        self.abdomen_poly = self.body_mdl.abdomen_Surf.return_polydata()
        self.wing_L0_poly = self.wing_mdl_L.return_polydata_mem0()
        self.wing_L1_poly = self.wing_mdl_L.return_polydata_mem1()
        self.wing_L2_poly = self.wing_mdl_L.return_polydata_mem2()
        self.wing_L3_poly = self.wing_mdl_L.return_polydata_mem3()
        self.wing_R0_poly = self.wing_mdl_R.return_polydata_mem0()
        self.wing_R1_poly = self.wing_mdl_R.return_polydata_mem1()
        self.wing_R2_poly = self.wing_mdl_R.return_polydata_mem2()
        self.wing_R3_poly = self.wing_mdl_R.return_polydata_mem3()

        # Compute wing length
        wing_L_bounds = self.wing_L0_poly.GetPoints().GetBounds()
        self.R_fly = wing_L_bounds[3]

        # compute mass, cg, Inertia tensor:
        self.head_m,     self.head_cg,      self.head_I,      self.head_S    = self.voxelize(self.head_poly   ,self.dx,self.scale[0],self.s_head,self.body_density,tol=1.0e-3)
        self.thorax_m,     self.thorax_cg,  self.thorax_I,  self.thorax_S     = self.voxelize(self.thorax_poly ,self.dx,self.scale[1],self.s_thorax,self.body_density,tol=1.0e-3)
        self.abdomen_m, self.abdomen_cg, self.abdomen_I, self.abdomen_S = self.voxelize(self.abdomen_poly,self.dx,self.scale[2],self.s_abdomen,self.body_density,tol=1.0e-3)
        self.wing_L0_m, self.wing_L0_cg, self.wing_L0_I, self.wing_L0_S = self.voxelize(self.wing_L0_poly,self.dx,self.scale[3],self.s_wing_L,self.wing_density,tol=1.0e-3)
        self.wing_L1_m, self.wing_L1_cg, self.wing_L1_I, self.wing_L1_S = self.voxelize(self.wing_L1_poly,self.dx,self.scale[3],self.s_wing_L,self.wing_density,tol=1.0e-3)
        self.wing_L2_m, self.wing_L2_cg, self.wing_L2_I, self.wing_L2_S = self.voxelize(self.wing_L2_poly,self.dx,self.scale[3],self.s_wing_L,self.wing_density,tol=1.0e-3)
        self.wing_L3_m, self.wing_L3_cg, self.wing_L3_I, self.wing_L3_S = self.voxelize(self.wing_L3_poly,self.dx,self.scale[3],self.s_wing_L,self.wing_density,tol=1.0e-3)
        self.wing_R0_m, self.wing_R0_cg, self.wing_R0_I, self.wing_R0_S = self.voxelize(self.wing_R0_poly,self.dx,self.scale[4],self.s_wing_R,self.wing_density,tol=1.0e-3)
        self.wing_R1_m, self.wing_R1_cg, self.wing_R1_I, self.wing_R1_S = self.voxelize(self.wing_R1_poly,self.dx,self.scale[4],self.s_wing_R,self.wing_density,tol=1.0e-3)
        self.wing_R2_m, self.wing_R2_cg, self.wing_R2_I, self.wing_R2_S = self.voxelize(self.wing_R2_poly,self.dx,self.scale[4],self.s_wing_R,self.wing_density,tol=1.0e-3)
        self.wing_R3_m, self.wing_R3_cg, self.wing_R3_I, self.wing_R3_S = self.voxelize(self.wing_R3_poly,self.dx,self.scale[4],self.s_wing_R,self.wing_density,tol=1.0e-3)

        # body mass
        self.body_mass = self.head_m+self.thorax_m+self.abdomen_m
        self.body_I = self.head_I+self.thorax_I+self.abdomen_I

        # combine wing inertia matrices to a rigid wing:
        self.wing_L_m  = self.wing_L0_m+self.wing_L1_m+self.wing_L2_m+self.wing_L3_m
        self.wing_L_cg = (self.wing_L0_cg*self.wing_L0_m+self.wing_L1_cg*self.wing_L1_m+self.wing_L2_cg*self.wing_L2_m+self.wing_L3_cg*self.wing_L3_m)/self.wing_L_m
        self.wing_L_I  = self.wing_L0_I+self.wing_L1_I+self.wing_L2_I+self.wing_L3_I
        self.wing_R_m  = self.wing_R0_m+self.wing_R1_m+self.wing_R2_m+self.wing_R3_m
        self.wing_R_cg = (self.wing_R0_cg*self.wing_R0_m+self.wing_R1_cg*self.wing_R1_m+self.wing_R2_cg*self.wing_R2_m+self.wing_R3_cg*self.wing_R3_m)/self.wing_R_m
        self.wing_R_I  = self.wing_R0_I+self.wing_R1_I+self.wing_R2_I+self.wing_R3_I

        self.joint_head = np.array([[0.55],[0.0],[0.42]])*self.scale[1]
        self.joint_abdomen = np.array([[0.0],[0.0],[-0.1]])*self.scale[1]
        self.joint_L = np.array([[0.0],[0.5],[0.0]])*self.scale[1]
        self.joint_R = np.array([[0.0],[-0.5],[0.0]])*self.scale[1]

        R_head = self.comp_R(self.s_head[:4])
        R_abdomen = self.comp_R(self.s_abdomen[:4])
        #self.body_cg = (self.head_m*(np.dot(R_head,self.head_cg)+self.joint_head)+self.thorax_m*self.thorax_cg+self.abdomen_m*(np.dot(R_abdomen,self.abdomen_cg)+self.joint_abdomen))/self.body_mass
        #self.body_cg = (self.head_m*self.joint_head+self.abdomen_m*self.joint_abdomen)/(self.head_m+self.abdomen_m)
        #self.body_cg = np.array([-0.2,0,-0.2])
        self.body_cg = np.array([0,0,-0.1])
        #self.body_cg = np.zeros(3)
        #self.body_cg[1] = 0

        # wing area
        self.wing_S          = self.wing_L0_S[0]+self.wing_L1_S[0]+self.wing_L2_S[0]+self.wing_L3_S[0]
        self.wing_S_xx         = self.wing_L0_S[1]+self.wing_L1_S[1]+self.wing_L2_S[1]+self.wing_L3_S[1]
        self.wing_S_yy         = self.wing_L0_S[2]+self.wing_L1_S[2]+self.wing_L2_S[2]+self.wing_L3_S[2]
        self.wing_S_xx_asym = self.wing_L0_S[3]+self.wing_L1_S[3]+self.wing_L2_S[3]+self.wing_L3_S[3]
        self.c_mean         = self.wing_S/self.R_fly

        # Compute M
        self.compute_M()

    def compute_M(self):
        self.Mb = np.zeros((6,6))
        self.Mb[0,0] = self.body_mass
        self.Mb[0,4] = self.body_mass*self.body_cg[2]
        self.Mb[0,5] = -self.body_mass*self.body_cg[1]
        self.Mb[1,1] = self.body_mass
        self.Mb[1,3] = -self.body_mass*self.body_cg[2]
        self.Mb[1,5] = self.body_mass*self.body_cg[0]
        self.Mb[2,2] = self.body_mass
        self.Mb[2,3] = self.body_mass*self.body_cg[1]
        self.Mb[2,4] = -self.body_mass*self.body_cg[0]
        self.Mb[3,1] = -self.body_mass*self.body_cg[2]
        self.Mb[3,2] = self.body_mass*self.body_cg[1]
        self.Mb[3,3] = self.body_I[0,0]
        self.Mb[3,4] = self.body_I[0,1]
        self.Mb[3,5] = self.body_I[0,2]
        self.Mb[4,0] = self.body_mass*self.body_cg[2]
        self.Mb[4,2] = -self.body_mass*self.body_cg[0]
        self.Mb[4,3] = self.body_I[1,0]
        self.Mb[4,4] = self.body_I[1,1]
        self.Mb[4,5] = self.body_I[1,2]
        self.Mb[5,0] = -self.body_mass*self.body_cg[1]
        self.Mb[5,1] = self.body_mass*self.body_cg[0]
        self.Mb[5,3] = self.body_I[2,0]
        self.Mb[5,4] = self.body_I[2,1]
        self.Mb[5,5] = self.body_I[2,2]

        self.MwL = np.zeros((6,6))
        self.MwL[0,0] = self.wing_L_m
        self.MwL[0,4] = self.wing_L_cg[2]*self.wing_L_m
        self.MwL[0,5] = -self.wing_L_cg[1]*self.wing_L_m
        self.MwL[1,1] = self.wing_L_m
        self.MwL[1,3] = -self.wing_L_cg[2]*self.wing_L_m
        self.MwL[1,5] = self.wing_L_cg[0]*self.wing_L_m
        self.MwL[2,2] = self.wing_L_m
        self.MwL[2,3] = self.wing_L_cg[1]*self.wing_L_m
        self.MwL[2,4] = -self.wing_L_cg[0]*self.wing_L_m
        self.MwL[3,1] = -self.wing_L_cg[2]*self.wing_L_m
        self.MwL[3,2] = self.wing_L_cg[1]*self.wing_L_m
        self.MwL[3,3] = self.wing_L_I[0,0]
        self.MwL[3,4] = self.wing_L_I[0,1]
        self.MwL[3,5] = self.wing_L_I[0,2]
        self.MwL[4,0] = self.wing_L_cg[2]*self.wing_L_m
        self.MwL[4,2] = -self.wing_L_cg[0]*self.wing_L_m
        self.MwL[4,3] = self.wing_L_I[1,0]
        self.MwL[4,4] = self.wing_L_I[1,1]
        self.MwL[4,5] = self.wing_L_I[1,2]
        self.MwL[5,0] = -self.wing_L_cg[1]*self.wing_L_m
        self.MwL[5,1] = self.wing_L_cg[0]*self.wing_L_m
        self.MwL[5,3] = self.wing_L_I[2,0]
        self.MwL[5,4] = self.wing_L_I[2,1]
        self.MwL[5,5] = self.wing_L_I[2,2]

        self.MwR = np.zeros((6,6))
        self.MwR[0,0] = self.wing_R_m
        self.MwR[0,4] = self.wing_R_cg[2]*self.wing_R_m
        self.MwR[0,5] = -self.wing_R_cg[1]*self.wing_R_m
        self.MwR[1,1] = self.wing_R_m
        self.MwR[1,3] = -self.wing_R_cg[2]*self.wing_R_m
        self.MwR[1,5] = self.wing_R_cg[0]*self.wing_R_m
        self.MwR[2,2] = self.wing_R_m
        self.MwR[2,3] = self.wing_R_cg[1]*self.wing_R_m
        self.MwR[2,4] = -self.wing_R_cg[0]*self.wing_R_m
        self.MwR[3,1] = -self.wing_R_cg[2]*self.wing_R_m
        self.MwR[3,2] = self.wing_R_cg[1]*self.wing_R_m
        self.MwR[3,3] = self.wing_R_I[0,0]
        self.MwR[3,4] = self.wing_R_I[0,1]
        self.MwR[3,5] = self.wing_R_I[0,2]
        self.MwR[4,0] = self.wing_R_cg[2]*self.wing_R_m
        self.MwR[4,2] = -self.wing_R_cg[0]*self.wing_R_m
        self.MwR[4,3] = self.wing_R_I[1,0]
        self.MwR[4,4] = self.wing_R_I[1,1]
        self.MwR[4,5] = self.wing_R_I[1,2]
        self.MwR[5,0] = -self.wing_R_cg[1]*self.wing_R_m
        self.MwR[5,1] = self.wing_R_cg[0]*self.wing_R_m
        self.MwR[5,3] = self.wing_R_I[2,0]
        self.MwR[5,4] = self.wing_R_I[2,1]
        self.MwR[5,5] = self.wing_R_I[2,2]

    def voxelize(self,poly_in,dx,scale_in,s_in,density_in,tol=0):
        # Construct the mesh grid from shape
        bnds = poly_in.GetBounds()
        nx = int(np.floor((bnds[1]-bnds[0])/dx)+1)
        ny = int(np.floor((bnds[3]-bnds[2])/dx)+1)
        nz = int(np.floor((bnds[5]-bnds[4])/dx)+1)

        gridx,gridy,gridz = np.meshgrid(np.linspace(bnds[0],bnds[1],num=nx),np.linspace(bnds[2],bnds[3],num=ny),np.linspace(bnds[4],bnds[5],num=nz))

        # Create polydata
        vtk_points = vtk.vtkPoints()
        for point in zip(gridx.flatten(), gridy.flatten(), gridz.flatten()):
            vtk_points.InsertNextPoint(point)
        points_polydata = vtk.vtkPolyData()
        points_polydata.SetPoints(vtk_points)

         # Compute enclosed points
        enclosed_pts = vtk.vtkSelectEnclosedPoints()
        enclosed_pts.SetInputData(points_polydata)
        enclosed_pts.SetTolerance(tol)
        enclosed_pts.SetSurfaceData(poly_in)
        #enclosed_pts.SetCheckSurface(1)
        enclosed_pts.Update()
        inside_points = enclosed_pts.GetOutput().GetPointData().GetArray("SelectedPoints")
        enclosed_pts.ReleaseDataFlagOn()
        enclosed_pts.Complete()

        # Convert result as a numpy array
        inside_array = vtk_to_numpy(inside_points).reshape(ny, nx, nz)
        inside_array = np.swapaxes(inside_array, 1, 0)

        dx_2 = dx*dx
        dx_3 = dx*dx*dx

        # compute cg
        cg = np.zeros((3,1))
        Inertia = np.zeros((3,3))
        S2 = np.zeros(4)
        cntr = 0
        for i in range(ny):
            for j in range(nx):
                for k in range(nz):
                    if inside_array[j][i][k]>0:
                        cg[0] += gridx[i][j][k]
                        cg[1] += gridy[i][j][k]
                        cg[2] += gridz[i][j][k]
                        Inertia[0,0] += (gridy[i][j][k]**2+gridz[i][j][k]**2)
                        Inertia[0,1] += -gridx[i][j][k]*gridy[i][j][k]
                        Inertia[0,2] += -gridx[i][j][k]*gridz[i][j][k]
                        Inertia[1,0] += -gridx[i][j][k]*gridy[i][j][k]
                        Inertia[1,1] += (gridx[i][j][k]**2+gridz[i][j][k]**2)
                        Inertia[1,2] += -gridy[i][j][k]*gridz[i][j][k]
                        Inertia[2,0] += -gridx[i][j][k]*gridz[i][j][k]
                        Inertia[2,1] += -gridy[i][j][k]*gridz[i][j][k]
                        Inertia[2,2] += (gridx[i][j][k]**2+gridy[i][j][k]**2)
                        cntr += 1
                        S2[0] += dx_2                                         # S
                        S2[1] += dx_2*gridx[i][j][k]**2                     # S_xx
                        S2[2] += dx_2*gridy[i][j][k]**2                     # S_yy
                        S2[3] += dx_2*gridx[i][j][k]*np.abs(gridx[i][j][k]) # S_x|x|
        mass = density_in*dx_3*cntr*scale_in**3
        Inertia = density_in*dx_3*scale_in**3*Inertia
        cg = cg/(cntr*1.0)*scale_in
        cg_cross = np.array([[0.0, -cg[2], cg[1]],[cg[2],0.0,-cg[0]],[-cg[1],cg[0],0.0]],dtype=np.double)
        mcc = mass*np.dot(cg_cross,cg_cross)
        Inertia = Inertia-mcc

        # Apply Rotation matrix on cg and Inertia:
        R_mat = self.comp_R(s_in[:4])
        cg_rotated = np.dot(np.transpose(R_mat),cg)
        Inertia_rotated = np.dot(np.transpose(R_mat),np.dot(Inertia,R_mat))

        return mass, cg_rotated, Inertia_rotated, S2

    def LegendrePolynomials(self,N_pts,N_pol,n_deriv):
        L_basis = np.zeros((N_pts,N_pol,n_deriv))
        x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
        for i in range(n_deriv):
            if i==0:
                # Legendre basis:
                for n in range(N_pol):
                    if n==0:
                        L_basis[:,n,i] = 1.0
                    elif n==1:
                        L_basis[:,n,i] = x_basis
                    else:
                        for k in range(n+1):
                            L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(scipy.special.binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
            else:
                # Derivatives:
                for n in range(N_pol):
                    if n>=i:
                        L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
        return L_basis

    def TemporalBC(self,a_c,N_pol,N_const):
        X_Legendre = self.LegendrePolynomials(100,N_pol,N_const)
        trace = np.dot(X_Legendre[:,:,0],a_c)
        b_L = np.zeros(9)
        b_L[0:4] = trace[-5:-1]
        b_L[4] = 0.5*(trace[0]+trace[-1])
        b_L[5:9] = trace[1:5]
        b_R = np.zeros(9)
        b_R[0:4] = trace[-5:-1]
        b_R[4] = 0.5*(trace[0]+trace[-1])
        b_R[5:9] = trace[1:5]
        c_per = self.LegendreFit(trace,b_L,b_R,N_pol,N_const)
        return c_per

    def LegendreBC(self,a_prev,a_now,a_next,N_pol,N_const):
        X_Legendre = self.LegendrePolynomials(30,N_pol,N_const)
        trace_prev = np.dot(X_Legendre[:,:,0],a_prev)
        trace_now  = np.dot(X_Legendre[:,:,0],a_now)
        trace_next = np.dot(X_Legendre[:,:,0],a_next)
        b_L = np.zeros(9)
        b_L[0:4] = trace_prev[-5:-1]
        b_L[4] = 0.5*(trace_now[0]+trace_prev[-1])
        b_L[5:9] = trace_now[1:5]
        b_R = np.zeros(9)
        b_R[0:4] = trace_now[-5:-1]
        b_R[4] = 0.5*(trace_next[0]+trace_now[-1])
        b_R[5:9] = trace_next[1:5]
        c_per = self.LegendreFit(trace_now,b_L,b_R,N_pol,N_const)
        return c_per

    def LegendreFit(self,trace_in,b1_in,b2_in,N_pol,N_const):
        N_pts = trace_in.shape[0]
        X_Legendre = self.LegendrePolynomials(N_pts,N_pol,N_const)
        A = X_Legendre[:,:,0]
        B = np.zeros((2*N_const,N_pol))
        # data points:
        b = np.transpose(trace_in)
        # restriction vector (add zeros to smooth the connection!!!!!)
        d = np.zeros(2*N_const)
        d_gradient_1 = b1_in
        d_gradient_2 = b2_in
        for j in range(N_const):
            d[j] = d_gradient_1[4-j]*np.power(N_pts/2.0,j)
            d[N_const+j] = d_gradient_2[4-j]*np.power(N_pts/2.0,j)
            d_gradient_1 = np.diff(d_gradient_1)
            d_gradient_2 = np.diff(d_gradient_2)
            B[j,:]             = np.transpose(X_Legendre[0,:,j])
            B[N_const+j,:]     = np.transpose(X_Legendre[-1,:,j])
        # Restricted least-squares fit:
        ATA = np.dot(np.transpose(A),A)
        ATA_inv = np.linalg.inv(ATA)
        AT = np.transpose(A)
        BT = np.transpose(B)
        BATABT     = np.dot(B,np.dot(ATA_inv,BT))
        c_ls     = np.linalg.solve(ATA,np.dot(AT,b))
        c_rls     = c_ls-np.dot(ATA_inv,np.dot(BT,np.linalg.solve(BATABT,np.dot(B,c_ls)-d)))
        return c_rls

    def q_mult(self,qA,qB):
        QA = np.array([[qA[0],-qA[1],-qA[2],-qA[3]],
            [qA[1],qA[0],-qA[3],qA[2]],
            [qA[2],qA[3],qA[0],-qA[1]],
            [qA[3],-qA[2],qA[1],qA[0]]])
        qC = np.dot(QA,qB)
        qC_norm = math.sqrt(pow(qC[0],2)+pow(qC[1],2)+pow(qC[2],2)+pow(qC[3],2))
        if qC_norm>0.01:
            qC /= qC_norm
        else:
            qC = np.array([1.0,0.0,0.0,0.0])
        return qC

    def comp_R(self,q):
        R = np.array([[2*pow(q[0],2)-1+2*pow(q[1],2), 2*q[1]*q[2]-2*q[0]*q[3], 2*q[1]*q[3]+2*q[0]*q[2]],
            [2*q[1]*q[2]+2*q[0]*q[3], 2*pow(q[0],2)-1+2*pow(q[2],2), 2*q[2]*q[3]-2*q[0]*q[1]],
            [2*q[1]*q[3]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], 2*pow(q[0],2)-1+2*pow(q[3],2)]])
        return R

    def quat_mat(self,s_in):
        q0 = np.squeeze(s_in[0])
        q1 = np.squeeze(s_in[1])
        q2 = np.squeeze(s_in[2])
        q3 = np.squeeze(s_in[3])
        tx = np.squeeze(s_in[4])
        ty = np.squeeze(s_in[5])
        tz = np.squeeze(s_in[6])
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2, tx],
            [2*q1*q2+2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3-2*q0*q1, ty],
            [2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        return M

    def set_srf_angle(self,beta_in):
        self.beta = beta_in

    def set_body_motion(self,vwb):
        self.v_b = vwb[:3]
        self.w_b = vwb[3:]

    def set_wing_motion(self,f_in,a_theta_L_in,a_eta_L_in,a_phi_L_in,a_xi_L_in,a_theta_R_in,a_eta_R_in,a_phi_R_in,a_xi_R_in,n_pts):

        self.freq  = f_in*1.0
        self.N_pts = n_pts
        self.dt    = 1.0/(f_in*n_pts)

        # Legendre coefficients:
        self.a_theta_L     = a_theta_L_in
        self.a_eta_L     = a_eta_L_in
        self.a_phi_L     = a_phi_L_in
        self.a_xi_L     = a_xi_L_in
        self.a_theta_R     = a_theta_R_in
        self.a_eta_R     = a_eta_R_in
        self.a_phi_R     = a_phi_R_in
        self.a_xi_R     = a_xi_R_in

        # Create tip traces:
        X_theta = self.LegendrePolynomials(self.N_pts,self.N_pol_theta,3)
        X_eta     = self.LegendrePolynomials(self.N_pts,self.N_pol_eta,3)
        X_phi     = self.LegendrePolynomials(self.N_pts,self.N_pol_phi,3)
        X_xi     = self.LegendrePolynomials(self.N_pts,self.N_pol_xi,3)

        # Wing kinematic angles:
        self.theta_L      = np.dot(X_theta[:,:,0],self.a_theta_L)
        self.eta_L          = np.dot(X_eta[:,:,0],self.a_eta_L)
        self.phi_L          = np.dot(X_phi[:,:,0],self.a_phi_L)
        self.xi_L          = np.dot(X_xi[:,:,0],self.a_xi_L)
        self.theta_R      = np.dot(X_theta[:,:,0],self.a_theta_R)
        self.eta_R          = np.dot(X_eta[:,:,0],self.a_eta_R)
        self.phi_R          = np.dot(X_phi[:,:,0],self.a_phi_R)
        self.xi_R          = np.dot(X_xi[:,:,0],self.a_xi_R)
        self.theta_dot_L = np.dot(X_theta[:,:,1],self.a_theta_L)*self.freq
        self.eta_dot_L   = np.dot(X_eta[:,:,1],self.a_eta_L)*self.freq
        self.phi_dot_L   = np.dot(X_phi[:,:,1],self.a_phi_L)*self.freq
        self.xi_dot_L      = np.dot(X_xi[:,:,1],self.a_xi_L)*self.freq
        self.theta_dot_R = np.dot(X_theta[:,:,1],self.a_theta_R)*self.freq
        self.eta_dot_R   = np.dot(X_eta[:,:,1],self.a_eta_R)*self.freq
        self.phi_dot_R   = np.dot(X_phi[:,:,1],self.a_phi_R)*self.freq
        self.xi_dot_R      = np.dot(X_xi[:,:,1],self.a_xi_R)*self.freq
        self.theta_ddot_L = np.dot(X_theta[:,:,2],self.a_theta_L)*self.freq**2
        self.eta_ddot_L   = np.dot(X_eta[:,:,2],self.a_eta_L)*self.freq**2
        self.phi_ddot_L   = np.dot(X_phi[:,:,2],self.a_phi_L)*self.freq**2
        self.xi_ddot_L       = np.dot(X_xi[:,:,2],self.a_xi_L)*self.freq**2
        self.theta_ddot_R = np.dot(X_theta[:,:,2],self.a_theta_R)*self.freq**2
        self.eta_ddot_R   = np.dot(X_eta[:,:,2],self.a_eta_R)*self.freq**2
        self.phi_ddot_R   = np.dot(X_phi[:,:,2],self.a_phi_R)*self.freq**2
        self.xi_ddot_R       = np.dot(X_xi[:,:,2],self.a_xi_R)*self.freq**2

        # SRF angle:
        q_beta       = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        R_beta       = self.comp_R(q_beta)

        q_90 = np.array([np.cos(-np.pi/4.0),0.0,np.sin(-np.pi/4.0),0.0])

        R_90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])

        # Compute angular velocities:
        self.R_Lw = np.zeros((3,3,self.N_pts))
        self.R_Rw = np.zeros((3,3,self.N_pts))
        self.w_Lw = np.zeros((3,self.N_pts))
        self.w_Rw = np.zeros((3,self.N_pts))
        self.w_dot_Lw = np.zeros((3,self.N_pts))
        self.w_dot_Rw = np.zeros((3,self.N_pts))

        for i in range(self.N_pts):
            # Left
            phi_L       = self.phi_L[i]
            theta_L   = -self.theta_L[i]
            eta_L     = self.eta_L[i]
            xi_L      = self.xi_L[i]
            q_phi_L   = np.array([np.cos(phi_L/2.0),np.sin(phi_L/2.0),0.0,0.0])
            q_theta_L = np.array([np.cos(theta_L/2.0),0.0,0.0,np.sin(theta_L/2.0)])
            q_eta_L   = np.array([np.cos(eta_L/2.0),0.0,np.sin(eta_L/2.0),0.0])
            phi_dot_L_vec = np.array([[self.phi_dot_L[i]],[0.0],[0.0]])
            theta_dot_L_vec = np.array([[0.0],[0.0],[-self.theta_dot_L[i]]])
            eta_dot_L_vec = np.array([[0.0],[self.eta_dot_L[i]],[0.0]])
            xi_dot_L_vec = np.array([[0.0],[self.xi_dot_L[i]],[0.0]])
            phi_ddot_L_vec = np.array([[self.phi_ddot_L[i]],[0.0],[0.0]])
            theta_ddot_L_vec = np.array([[0.0],[0.0],[-self.theta_ddot_L[i]]])
            eta_ddot_L_vec = np.array([[0.0],[self.eta_ddot_L[i]],[0.0]])
            xi_ddot_L_vec = np.array([[0.0],[self.xi_ddot_L[i]],[0.0]])
            #q_L = self.q_mult(q_beta,self.q_mult(q_phi_L,self.q_mult(q_theta_L,q_eta_L)))
            q_L = self.q_mult(q_phi_L,self.q_mult(q_theta_L,q_eta_L))
            R_L = np.transpose(self.comp_R(q_L))
            self.R_Lw[:,:,i] = R_L
            self.w_Lw[:,i] = np.squeeze(np.dot(np.transpose(R_L),np.dot(R_90,self.w_b))+np.dot(self.comp_R(self.q_mult(q_eta_L,q_theta_L)),phi_dot_L_vec)+np.dot(self.comp_R(q_eta_L),theta_dot_L_vec)+eta_dot_L_vec)
            self.w_dot_Lw[:,i] = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta_L,q_theta_L)),phi_ddot_L_vec)+np.dot(self.comp_R(q_eta_L),theta_ddot_L_vec)+eta_ddot_L_vec)
            # Right
            phi_R       = -self.phi_R[i]
            theta_R   = self.theta_R[i]
            eta_R     = self.eta_R[i]
            xi_R      = self.xi_R[i]
            q_phi_R   = np.array([np.cos(phi_R/2.0),np.sin(phi_R/2.0),0.0,0.0])
            q_theta_R = np.array([np.cos(theta_R/2.0),0.0,0.0,np.sin(theta_R/2.0)])
            q_eta_R   = np.array([np.cos(eta_R/2.0),0.0,np.sin(eta_R/2.0),0.0])
            phi_dot_R_vec = np.array([[-self.phi_dot_R[i]],[0.0],[0.0]])
            theta_dot_R_vec = np.array([[0.0],[0.0],[self.theta_dot_R[i]]])
            eta_dot_R_vec = np.array([[0.0],[self.eta_dot_R[i]],[0.0]])
            xi_dot_R_vec = np.array([[0.0],[self.xi_dot_R[i]],[0.0]])
            phi_ddot_R_vec = np.array([[-self.phi_ddot_R[i]],[0.0],[0.0]])
            theta_ddot_R_vec = np.array([[0.0],[0.0],[self.theta_ddot_R[i]]])
            eta_ddot_R_vec = np.array([[0.0],[self.eta_ddot_R[i]],[0.0]])
            xi_ddot_R_vec = np.array([[0.0],[self.xi_ddot_R[i]],[0.0]])
            #q_R = self.q_mult(q_beta,self.q_mult(q_phi_R,self.q_mult(q_theta_R,q_eta_R)))
            q_R = self.q_mult(q_phi_R,self.q_mult(q_theta_R,q_eta_R))
            R_R = np.transpose(self.comp_R(q_R))
            self.R_Rw[:,:,i] = R_R
            self.w_Rw[:,i] = np.squeeze(np.dot(np.transpose(R_R),np.dot(R_90,self.w_b))+np.dot(self.comp_R(self.q_mult(q_eta_R,q_theta_R)),phi_dot_R_vec)+np.dot(self.comp_R(q_eta_R),theta_dot_R_vec)+eta_dot_R_vec)
            self.w_dot_Rw[:,i] = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta_R,q_theta_R)),phi_ddot_R_vec)+np.dot(self.comp_R(q_eta_R),theta_ddot_R_vec)+eta_ddot_R_vec)

    def set_wing_motion2(self,f_in,theta_L_in,eta_L_in,phi_L_in,xi_L_in,theta_R_in,eta_R_in,phi_R_in,xi_R_in):

        self.freq  = f_in*1.0
        self.N_pts = theta_L_in.shape[0]
        self.dt    = 1.0/(f_in*self.N_pts)

        # Wing kinematic angles:
        self.theta_L       = theta_L_in
        self.eta_L           = eta_L_in
        self.phi_L           = phi_L_in
        self.xi_L           = xi_L_in
        self.theta_R       = theta_R_in
        self.eta_R           = eta_R_in
        self.phi_R           = phi_R_in
        self.xi_R           = xi_R_in
        self.theta_dot_L  = np.squeeze(np.gradient(self.theta_L)*1.0/self.dt)
        self.eta_dot_L    = np.squeeze(np.gradient(self.eta_L)*1.0/self.dt)
        self.phi_dot_L    = np.squeeze(np.gradient(self.phi_L)*1.0/self.dt)
        self.xi_dot_L       = np.squeeze(np.gradient(self.xi_L)*1.0/self.dt)
        self.theta_dot_R  = np.squeeze(np.gradient(self.theta_R)*1.0/self.dt)
        self.eta_dot_R    = np.squeeze(np.gradient(self.eta_R)*1.0/self.dt)
        self.phi_dot_R    = np.squeeze(np.gradient(self.phi_R)*1.0/self.dt)
        self.xi_dot_R       = np.squeeze(np.gradient(self.xi_R)*1.0/self.dt)
        self.theta_ddot_L = np.squeeze(np.gradient(self.theta_dot_L)*1.0/self.dt)
        self.eta_ddot_L   = np.squeeze(np.gradient(self.eta_dot_L)*1.0/self.dt)
        self.phi_ddot_L   = np.squeeze(np.gradient(self.phi_dot_L)*1.0/self.dt)
        self.xi_ddot_L       = np.squeeze(np.gradient(self.xi_dot_L)*1.0/self.dt)
        self.theta_ddot_R = np.squeeze(np.gradient(self.theta_dot_R)*1.0/self.dt)
        self.eta_ddot_R   = np.squeeze(np.gradient(self.eta_dot_R)*1.0/self.dt)
        self.phi_ddot_R   = np.squeeze(np.gradient(self.phi_dot_R)*1.0/self.dt)
        self.xi_ddot_R       = np.squeeze(np.gradient(self.xi_dot_R)*1.0/self.dt)

        # SRF angle:
        q_beta       = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        R_beta       = self.comp_R(q_beta)

        q_90 = np.array([np.cos(-np.pi/4.0),0.0,np.sin(-np.pi/4.0),0.0])

        R_90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])

        # Compute angular velocities:
        self.R_Lw = np.zeros((3,3,self.N_pts))
        self.R_Rw = np.zeros((3,3,self.N_pts))
        self.w_Lw = np.zeros((3,self.N_pts))
        self.w_Rw = np.zeros((3,self.N_pts))
        self.w_dot_Lw = np.zeros((3,self.N_pts))
        self.w_dot_Rw = np.zeros((3,self.N_pts))

        for i in range(self.N_pts):
            # Left
            phi_L       = self.phi_L[i]
            theta_L   = -self.theta_L[i]
            eta_L     = self.eta_L[i]
            xi_L      = self.xi_L[i]
            q_phi_L   = np.array([np.cos(phi_L/2.0),np.sin(phi_L/2.0),0.0,0.0])
            q_theta_L = np.array([np.cos(theta_L/2.0),0.0,0.0,np.sin(theta_L/2.0)])
            q_eta_L   = np.array([np.cos(eta_L/2.0),0.0,np.sin(eta_L/2.0),0.0])
            phi_dot_L_vec = np.array([[self.phi_dot_L[i]],[0.0],[0.0]])
            theta_dot_L_vec = np.array([[0.0],[0.0],[-self.theta_dot_L[i]]])
            eta_dot_L_vec = np.array([[0.0],[self.eta_dot_L[i]],[0.0]])
            xi_dot_L_vec = np.array([[0.0],[self.xi_dot_L[i]],[0.0]])
            phi_ddot_L_vec = np.array([[self.phi_ddot_L[i]],[0.0],[0.0]])
            theta_ddot_L_vec = np.array([[0.0],[0.0],[-self.theta_ddot_L[i]]])
            eta_ddot_L_vec = np.array([[0.0],[self.eta_ddot_L[i]],[0.0]])
            xi_ddot_L_vec = np.array([[0.0],[self.xi_ddot_L[i]],[0.0]])
            #q_L = self.q_mult(q_beta,self.q_mult(q_phi_L,self.q_mult(q_theta_L,q_eta_L)))
            q_L = self.q_mult(q_phi_L,self.q_mult(q_theta_L,q_eta_L))
            R_L = np.transpose(self.comp_R(q_L))
            self.R_Lw[:,:,i] = R_L
            self.w_Lw[:,i] = np.squeeze(np.dot(np.transpose(R_L),np.dot(R_90,self.w_b))+np.dot(self.comp_R(self.q_mult(q_eta_L,q_theta_L)),phi_dot_L_vec)+np.dot(self.comp_R(q_eta_L),theta_dot_L_vec)+eta_dot_L_vec)
            self.w_dot_Lw[:,i] = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta_L,q_theta_L)),phi_ddot_L_vec)+np.dot(self.comp_R(q_eta_L),theta_ddot_L_vec)+eta_ddot_L_vec)
            # Right
            phi_R       = -self.phi_R[i]
            theta_R   = self.theta_R[i]
            eta_R     = self.eta_R[i]
            xi_R      = self.xi_R[i]
            q_phi_R   = np.array([np.cos(phi_R/2.0),np.sin(phi_R/2.0),0.0,0.0])
            q_theta_R = np.array([np.cos(theta_R/2.0),0.0,0.0,np.sin(theta_R/2.0)])
            q_eta_R   = np.array([np.cos(eta_R/2.0),0.0,np.sin(eta_R/2.0),0.0])
            phi_dot_R_vec = np.array([[-self.phi_dot_R[i]],[0.0],[0.0]])
            theta_dot_R_vec = np.array([[0.0],[0.0],[self.theta_dot_R[i]]])
            eta_dot_R_vec = np.array([[0.0],[self.eta_dot_R[i]],[0.0]])
            xi_dot_R_vec = np.array([[0.0],[self.xi_dot_R[i]],[0.0]])
            phi_ddot_R_vec = np.array([[-self.phi_ddot_R[i]],[0.0],[0.0]])
            theta_ddot_R_vec = np.array([[0.0],[0.0],[self.theta_ddot_R[i]]])
            eta_ddot_R_vec = np.array([[0.0],[self.eta_ddot_R[i]],[0.0]])
            xi_ddot_R_vec = np.array([[0.0],[self.xi_ddot_R[i]],[0.0]])
            #q_R = self.q_mult(q_beta,self.q_mult(q_phi_R,self.q_mult(q_theta_R,q_eta_R)))
            q_R = self.q_mult(q_phi_R,self.q_mult(q_theta_R,q_eta_R))
            R_R = np.transpose(self.comp_R(q_R))
            self.R_Rw[:,:,i] = R_R
            self.w_Rw[:,i] = np.squeeze(np.dot(np.transpose(R_R),np.dot(R_90,self.w_b))+np.dot(self.comp_R(self.q_mult(q_eta_R,q_theta_R)),phi_dot_R_vec)+np.dot(self.comp_R(q_eta_R),theta_dot_R_vec)+eta_dot_R_vec)
            self.w_dot_Rw[:,i] = np.squeeze(np.dot(self.comp_R(self.q_mult(q_eta_R,q_theta_R)),phi_ddot_R_vec)+np.dot(self.comp_R(q_eta_R),theta_ddot_R_vec)+eta_ddot_R_vec)

    def quasi_steady_FT(self):

        self.alpha_LR = np.zeros((2,self.N_pts))

        self.FT_qs_Lw = np.zeros((6,self.N_pts))
        self.FT_qs_Rw = np.zeros((6,self.N_pts))

        self.FT_qs_Lb = np.zeros((6,self.N_pts))
        self.FT_qs_Rb = np.zeros((6,self.N_pts))

        R_90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])

        U_L = np.zeros(3)
        U_R = np.zeros(3)

        for i in range(self.N_pts):

            U_vb_L = np.dot(np.transpose(self.R_Lw[:,:,i]),np.dot(R_90,self.v_b))

            U_L[0] = self.w_Lw[2,i]*-0.7+U_vb_L[0]
            U_L[1] = self.w_Lw[1,i]
            U_L[2] = self.w_Lw[0,i]*0.7+U_vb_L[2]

            alpha_L = np.arctan2(-U_L[2],U_L[0])

            self.alpha_LR[0,i] = alpha_L

            #x_cp_L = (0.82*np.abs(alpha_L)/np.pi+0.05)*self.c_mean
            x_cp_L = (0.82*np.abs(alpha_L)/np.pi-0.2)*self.c_mean

            CL_L = np.sign(alpha_L)*(0.225+1.58*np.sin(2.13*np.abs(alpha_L)-7.2/180.0*np.pi))
            CD_L = np.sign(alpha_L)*(1.92-1.55*np.cos(2.04*np.abs(alpha_L)-9.82/180.0*np.pi))
            CR_L = 2.08

            Lift_L  = 0.5*self.rho_air*CL_L*self.wing_S_yy*(U_L[0]**2+U_L[2]**2)
            Drag_L  = 0.5*self.rho_air*CD_L*self.wing_S_yy*(U_L[0]**2+U_L[2]**2)
            Rot_L     = CR_L*self.rho_air*(np.sqrt(self.wing_S_xx*self.wing_S_yy)*np.sqrt(U_L[0]**2+U_L[2]**2)*U_L[1]+self.wing_S_xx_asym*np.sign(U_L[1])*U_L[1]**2)

            self.FT_qs_Lw[0,i] = np.sin(alpha_L)*Lift_L-np.cos(alpha_L)*Drag_L
            self.FT_qs_Lw[1,i] = 0.0
            self.FT_qs_Lw[2,i] = np.cos(alpha_L)*Lift_L+np.sin(alpha_L)*Drag_L+Rot_L
            self.FT_qs_Lw[3,i] = 0.7*self.R_fly*self.FT_qs_Lw[2,i]
            self.FT_qs_Lw[4,i] = x_cp_L*self.FT_qs_Lw[2,i]
            self.FT_qs_Lw[5,i] = -0.7*self.R_fly*self.FT_qs_Lw[0,i]

            F_Lb = np.dot(np.transpose(R_90),np.dot(self.R_Lw[:,:,i],self.FT_qs_Lw[:3,i]))
            T_Lb = np.dot(np.transpose(R_90),np.dot(self.R_Lw[:,:,i],self.FT_qs_Lw[3:,i]))

            self.FT_qs_Lb[0,i] = F_Lb[0]
            self.FT_qs_Lb[1,i] = F_Lb[1]
            self.FT_qs_Lb[2,i] = F_Lb[2]
            self.FT_qs_Lb[3,i] = T_Lb[0]
            self.FT_qs_Lb[4,i] = T_Lb[1]
            self.FT_qs_Lb[5,i] = T_Lb[2]

            U_vb_R = np.dot(np.transpose(self.R_Rw[:,:,i]),np.dot(R_90,self.v_b))

            U_R[0] = self.w_Rw[2,i]*0.7+U_vb_R[0]
            U_R[1] = self.w_Rw[1,i]
            U_R[2] = self.w_Rw[0,i]*-0.7+U_vb_R[2]

            alpha_R = np.arctan2(-U_R[2],U_R[0])

            self.alpha_LR[1,i] = alpha_R

            #x_cp_R = (0.82*np.abs(alpha_R)/np.pi+0.05)*self.c_mean
            x_cp_R = (0.82*np.abs(alpha_R)/np.pi-0.2)*self.c_mean

            CL_R = np.sign(alpha_R)*(0.225+1.58*np.sin(2.13*np.abs(alpha_R)-7.2/180.0*np.pi))
            CD_R = np.sign(alpha_R)*(1.92-1.55*np.cos(2.04*np.abs(alpha_R)-9.82/180.0*np.pi))
            CR_R = 2.08

            Lift_R  = 0.5*self.rho_air*CL_R*self.wing_S_yy*(U_R[0]**2+U_R[2]**2)
            Drag_R  = 0.5*self.rho_air*CD_R*self.wing_S_yy*(U_R[0]**2+U_R[2]**2)
            Rot_R     = CR_R*self.rho_air*(np.sqrt(self.wing_S_xx*self.wing_S_yy)*np.sqrt(U_R[0]**2+U_R[2]**2)*U_R[1]+self.wing_S_xx_asym*np.sign(U_R[1])*U_R[1]**2)

            self.FT_qs_Rw[0,i] = np.sin(alpha_R)*Lift_R-np.cos(alpha_R)*Drag_R
            self.FT_qs_Rw[1,i] = 0.0
            self.FT_qs_Rw[2,i] = np.cos(alpha_R)*Lift_R+np.sin(alpha_R)*Drag_R+Rot_R
            self.FT_qs_Rw[3,i] = -0.7*self.R_fly*self.FT_qs_Rw[2,i]
            self.FT_qs_Rw[4,i] = x_cp_R*self.FT_qs_Rw[2,i]
            self.FT_qs_Rw[5,i] = 0.7*self.R_fly*self.FT_qs_Rw[0,i]

            F_Rb = np.dot(np.transpose(R_90),np.dot(self.R_Rw[:,:,i],self.FT_qs_Rw[:3,i]))
            T_Rb = np.dot(np.transpose(R_90),np.dot(self.R_Rw[:,:,i],self.FT_qs_Rw[3:,i]))

            self.FT_qs_Rb[0,i] = F_Rb[0]
            self.FT_qs_Rb[1,i] = F_Rb[1]
            self.FT_qs_Rb[2,i] = F_Rb[2]
            self.FT_qs_Rb[3,i] = T_Rb[0]
            self.FT_qs_Rb[4,i] = T_Rb[1]
            self.FT_qs_Rb[5,i] = T_Rb[2]

    def inertia_FT(self):
        # compute inertial forces and torques:

        self.FTI_acc_Lw = np.zeros((6,self.N_pts))
        self.FTI_vel_Lw = np.zeros((6,self.N_pts))
        self.FTI_acc_Lb = np.zeros((6,self.N_pts))
        self.FTI_vel_Lb = np.zeros((6,self.N_pts))

        self.FTI_acc_Rw = np.zeros((6,self.N_pts))
        self.FTI_vel_Rw = np.zeros((6,self.N_pts))
        self.FTI_acc_Rb = np.zeros((6,self.N_pts))
        self.FTI_vel_Rb = np.zeros((6,self.N_pts))

        R_90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])

        for i in range(self.N_pts):

            w_L_cross = np.array([[0.0,-self.w_Lw[2,i],self.w_Lw[1,i]],[self.w_Lw[2,i],0.0,-self.w_Lw[0,i]],[-self.w_Lw[1,i],self.w_Lw[0,i],0.0]])
            w_R_cross = np.array([[0.0,-self.w_Rw[2,i],self.w_Rw[1,i]],[self.w_Rw[2,i],0.0,-self.w_Rw[0,i]],[-self.w_Rw[1,i],self.w_Rw[0,i],0.0]])

            self.FTI_acc_Lw[:3,i] = np.dot(-self.MwL[:3,3:],self.w_dot_Lw[:,i])
            self.FTI_acc_Lw[3:,i] = np.dot(-self.MwL[3:,3:],self.w_dot_Lw[:,i])
            self.FTI_acc_Lb[:3,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Lw[:,:,i],self.FTI_acc_Lw[:3,i])))
            self.FTI_acc_Lb[3:,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Lw[:,:,i],self.FTI_acc_Lw[3:,i])))

            self.FTI_acc_Rw[:3,i] = np.dot(-self.MwR[:3,3:],self.w_dot_Rw[:,i])
            self.FTI_acc_Rw[3:,i] = np.dot(-self.MwR[3:,3:],self.w_dot_Rw[:,i])
            self.FTI_acc_Rb[:3,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Rw[:,:,i],self.FTI_acc_Rw[:3,i])))
            self.FTI_acc_Rb[3:,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Rw[:,:,i],self.FTI_acc_Rw[3:,i])))

            self.FTI_vel_Lw[:3,i] = np.squeeze(-self.wing_L_m*np.dot(w_L_cross,np.dot(w_L_cross,self.wing_L_cg)))
            self.FTI_vel_Lw[3:,i] = np.squeeze(-np.dot(w_L_cross,np.dot(self.wing_L_I,self.w_Lw[:,i])))
            self.FTI_vel_Lb[:3,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Lw[:,:,i],self.FTI_vel_Lw[:3,i])))
            self.FTI_vel_Lb[3:,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Lw[:,:,i],self.FTI_vel_Lw[3:,i])))

            self.FTI_vel_Rw[:3,i] = np.squeeze(-self.wing_R_m*np.dot(w_R_cross,np.dot(w_R_cross,self.wing_R_cg)))
            self.FTI_vel_Rw[3:,i] = np.squeeze(-np.dot(w_R_cross,np.dot(self.wing_R_I,self.w_Rw[:,i])))
            self.FTI_vel_Rb[:3,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Rw[:,:,i],self.FTI_vel_Rw[:3,i])))
            self.FTI_vel_Rb[3:,i] = np.squeeze(np.dot(np.transpose(R_90),np.dot(self.R_Rw[:,:,i],self.FTI_vel_Rw[3:,i])))


    def build_network(self):
        input_enc = Input(shape=(9,13,1))
        enc = GaussianNoise(0.05)(input_enc)
        enc = Conv2D(filters=64,kernel_size=(9,1),strides=(9,1),activation='selu')(enc)
        enc = Conv2D(filters=256,kernel_size=(1,13),strides=(1,13),activation='selu')(enc)
        encoded = Flatten()(enc)
        input_dec = Input(shape=(256,))
        dec = Dense(1024,activation='selu')(input_dec)
        decoded = Dense(80,activation='linear')(dec)
        encoder_model = Model(input_enc, encoded)
        decoder_model = Model(input_dec, decoded)
        auto_input = Input(shape=(9,13,1))
        encoded = encoder_model(auto_input)
        decoded = decoder_model(encoded)
        model = Model(auto_input, decoded)
        return model

    def load_network(self):
        # WBD: change
        # -----------------------------------------
        #print(cwd+'/weights')
        #os.chdir(cwd+'/weights')
        #weight_file = 'muscle_wing_weights_new.h5'
        # -----------------------------------------
        self.network = self.build_network()
        self.network.load_weights(self.weight_file)
        self.network.summary()
        print('network weigths loaded')

    def predict(self,X_in):
        prediction = self.network.predict(X_in)
        return prediction

    def Muscle_scale(self,X_in):
        X_out = X_in
        X_out[:,:12] = np.clip(X_in[:,:12],-0.5,1.5)
        X_out[:,12] = (np.clip(X_in[:,12],150.0,250.0)-150.0)/100.0
        return X_out

    def Muscle_scale_inverse(self,X_in):
        X_out = X_in
        X_out[:,:12] = X_in[:,:12]
        X_out[:,12] = X_in[:,12]*100.0+150.0
        return X_out

    def Wingkin_scale(self,X_in):
        X_out = (1.0/np.pi)*np.clip(X_in,-np.pi,np.pi)
        #X_out = np.clip(X_in,-np.pi,np.pi)
        return X_out

    def Wingkin_scale_inverse(self,X_in):
        X_out = X_in*np.pi
        return X_out

    def setup_MPC(self):
        model_type = 'discrete' # either 'discrete' or 'continuous'
        self.mpc_model = do_mpc.model.Model(model_type)

        self.M = self.Mb/self.Mb[0,0]
        print('M')
        print(self.M)
        print('')

        self.M_inv = np.linalg.inv(self.M)
        print('M inv')
        print(self.M_inv)
        print('')

        # Controls matrix (computed from RoboFly results):
        self.dFT_du = np.matrix([
            [-0.13798,  0.25996, -0.30252, -0.24198, -0.54039, -0.26602,  0.19141, -0.54956, -0.06172, -0.03087,  0.23443, -0.34096,
             -0.13798,  0.25996, -0.30252, -0.24198, -0.54039, -0.26602,  0.19141, -0.54956, -0.06172, -0.03087,  0.23443, -0.34096],
            [-0.10072,  0.32773, -0.02465, -0.15886, -0.14818,  0.04307,  0.49669,  0.14305, -0.14897, -0.15940,  0.46593,  0.15202,
              0.10072, -0.32773,  0.02465,  0.15886,  0.14818, -0.04307, -0.49669, -0.14305,  0.14897,  0.15940, -0.46593, -0.15202],
            [ 0.18469,  0.81548, -0.29041, -0.18003, -0.37618,  0.26172,  1.23383,  0.06462, -0.03146,  0.07927,  1.23604,  0.19249,
              0.18469,  0.81548, -0.29041, -0.18003, -0.37618,  0.26172,  1.23383,  0.06462, -0.03146,  0.07927,  1.23604,  0.19249],
            [ 0.07039,  0.42623, -0.24396, -0.12748, -0.25041,  0.10125,  0.53187, -0.03862, -0.03351, -0.00422,  0.54514,  0.11267,
             -0.07039, -0.42623,  0.24396,  0.12748,  0.25041, -0.10125, -0.53187,  0.03862,  0.03351,  0.00422, -0.54514, -0.11267],
            [-0.02709, -0.10261, -0.02198,  0.05901,  0.05636, -0.02760, -0.06135,  0.04134, -0.00637, -0.03596,  0.11300,  0.15851,
             -0.02709, -0.10261, -0.02198,  0.05901,  0.05636, -0.02760, -0.06135,  0.04134, -0.00637, -0.03596,  0.11300,  0.15851],
            [ 0.05316, -0.33061,  0.22744,  0.20070,  0.37587,  0.14049, -0.09744,  0.17975,  0.10784,  0.11541, -0.05937,  0.18306,
             -0.05316,  0.33061, -0.22744, -0.20070, -0.37587, -0.14049,  0.09744, -0.17975, -0.10784, -0.11541,  0.05937, -0.18306]])

        # muscle activation slopes
        self.muscle_coeffs = np.array([
            [ 1.00000000,  0.38294635,  0.08487301, -0.26850879, -0.14746883,  0.20532689,  0.18323582,  0.31902772, -0.15752858, -0.08646779,  0.37753297,  0.15545411],
            [ 0.12069261,  1.00000000, -0.04522912, -0.06616104, -0.01409848,  0.34046396,  0.07660047,  0.20211201, -0.08819629, -0.06520906,  0.24001988,  0.25200486],
            [ 0.09405810, -0.13910754,  1.00000000,  0.36078432,  0.36441537,  0.03816575, -0.26884987, -0.03253367,  0.21662994,  0.32773313, -0.07184409,  0.09465812],
            [-0.29969232,  0.05973252,  0.59502241,  1.00000000,  0.85791627,  0.40502641, -0.86009618, -0.16850873,  0.72542211,  0.73166629, -0.29231285,  0.04493139],
            [-0.19584931,  0.19147263,  0.42508704,  0.64591088,  1.00000000,  0.44961226, -0.78042855,  0.20465317,  0.56402615,  0.71486917, -0.15470505,  0.39624904],
            [ 0.05767091,  0.26415430,  0.04052168,  0.22984270,  0.22885451,  1.00000000, -0.22549753,  0.23029824,  0.11789540,  0.16739554,  0.17903193,  0.25080379],
            [ 0.23713989, -0.03408380, -0.34024917, -0.65283872, -0.78764487, -0.34712156,  1.00000000, -0.08028532, -0.71362593, -0.77562552,  0.26161918, -0.08966248],
            [ 0.07818664,  0.50355814, -0.00696217,  0.07420613,  0.22680787,  0.45860061, -0.06600282,  1.00000000, -0.08762214,  0.08643462,  0.29130090,  0.59450351],
            [-0.19958767, -0.15593995,  0.38975632,  0.76857650,  0.73307090,  0.16414442, -0.86181405, -0.29247957,  1.00000000,  0.89957931, -0.23618882, -0.25145880],
            [-0.10932595, -0.05062927,  0.36512142,  0.62042088,  0.71098788,  0.34403077, -0.79451684,  0.01609839,  0.78905980,  1.00000000,  0.00600789,  0.00553354],
            [ 0.29637255,  0.19824820, -0.07662596, -0.14693465, -0.14187301,  0.35026925,  0.07326336,  0.31765297, -0.08532741, -0.01508839,  1.00000000,  0.06399767],
            [-0.07549336,  0.40162006,  0.13736492,  0.17277185,  0.35913708,  0.33227760, -0.11332237,  0.57581652, -0.05777687,  0.07908884,  0.07254797,  1.00000000]])

        # Set variables:
        _vx = self.mpc_model.set_variable(var_type='_x', var_name='vx') # vx in body ref frame
        _vy = self.mpc_model.set_variable(var_type='_x', var_name='vy') # vy in body ref frame
        _vz = self.mpc_model.set_variable(var_type='_x', var_name='vz') # vz in body ref frame
        _wx = self.mpc_model.set_variable(var_type='_x', var_name='wx') # wx in body ref frame
        _wy = self.mpc_model.set_variable(var_type='_x', var_name='wy') # wy in body ref frame
        _wz = self.mpc_model.set_variable(var_type='_x', var_name='wz') # wz in body ref frame
        _q0 = self.mpc_model.set_variable(var_type='_x', var_name='q0') # q0 in inertial ref frame
        _qx = self.mpc_model.set_variable(var_type='_x', var_name='qx') # qx in inertial ref frame
        _qy = self.mpc_model.set_variable(var_type='_x', var_name='qy') # qy in inertial ref frame
        _qz = self.mpc_model.set_variable(var_type='_x', var_name='qz') # qz in inertial ref frame
        _px  = self.mpc_model.set_variable(var_type='_x', var_name='px') # px in inertial ref frame
        _py  = self.mpc_model.set_variable(var_type='_x', var_name='py') # py in inertial ref frame
        _pz  = self.mpc_model.set_variable(var_type='_x', var_name='pz') # pz in inertial ref frame

        self.x = [_vx,_vy,_vz,_wx,_wy,_wz,_q0,_qx,_qy,_qz,_px,_py,_pz]

        _u0L  = self.mpc_model.set_variable(var_type='_u', var_name='u0L')
        _u1L  = self.mpc_model.set_variable(var_type='_u', var_name='u1L')
        _u2L  = self.mpc_model.set_variable(var_type='_u', var_name='u2L')
        _u3L  = self.mpc_model.set_variable(var_type='_u', var_name='u3L')
        _u4L  = self.mpc_model.set_variable(var_type='_u', var_name='u4L')
        _u5L  = self.mpc_model.set_variable(var_type='_u', var_name='u5L')
        _u6L  = self.mpc_model.set_variable(var_type='_u', var_name='u6L')
        _u7L  = self.mpc_model.set_variable(var_type='_u', var_name='u7L')
        _u8L  = self.mpc_model.set_variable(var_type='_u', var_name='u8L')
        _u9L  = self.mpc_model.set_variable(var_type='_u', var_name='u9L')
        _u10L = self.mpc_model.set_variable(var_type='_u', var_name='u10L')
        _u11L = self.mpc_model.set_variable(var_type='_u', var_name='u11L')

        self.uL = [_u0L,_u1L,_u2L,_u3L,_u4L,_u5L,_u6L,_u7L,_u8L,_u9L,_u10L,_u11L]

        _u0R  = self.mpc_model.set_variable(var_type='_u', var_name='u0R')
        _u1R  = self.mpc_model.set_variable(var_type='_u', var_name='u1R')
        _u2R  = self.mpc_model.set_variable(var_type='_u', var_name='u2R')
        _u3R  = self.mpc_model.set_variable(var_type='_u', var_name='u3R')
        _u4R  = self.mpc_model.set_variable(var_type='_u', var_name='u4R')
        _u5R  = self.mpc_model.set_variable(var_type='_u', var_name='u5R')
        _u6R  = self.mpc_model.set_variable(var_type='_u', var_name='u6R')
        _u7R  = self.mpc_model.set_variable(var_type='_u', var_name='u7R')
        _u8R  = self.mpc_model.set_variable(var_type='_u', var_name='u8R')
        _u9R  = self.mpc_model.set_variable(var_type='_u', var_name='u9R')
        _u10R = self.mpc_model.set_variable(var_type='_u', var_name='u10R')
        _u11R = self.mpc_model.set_variable(var_type='_u', var_name='u11R')

        self.uR = [_u0R,_u1R,_u2R,_u3R,_u4R,_u5R,_u6R,_u7R,_u8R,_u9R,_u10R,_u11R]

        # Inertial damping
        dFT_dx_I = np.matrix([
            [0.0,0.0,0.0,     0.0, 0.0171,    0.0],
            [0.0,0.0,0.0,-0.03350,    0.0,-0.0316],
            [0.0,0.0,0.0,     0.0,-0.0107,    0.0],
            [0.0,0.0,0.0,-0.00611,    0.0,0.00571],
            [0.0,0.0,0.0,     0.0,0.00126,    0.0],
            [0.0,0.0,0.0,-0.01025,    0.0,0.00485]])

        # Aerodynamic damping
        dFT_dx_A = np.matrix([
            [-0.3588,    0.0, 0.0059,    0.0,-0.0022,    0.0],
            [     0.0,-0.2696,    0.0, 0.0022,    0.0, 0.1799],
            [-0.0025,    0.0,-0.0364,    0.0, 0.0093,    0.0],
            [    0.0,-0.0010,    0.0,-0.0185,    0.0, 0.0025],
            [-0.0010,    0.0,-0.0091,    0.0, 0.0232,    0.0],
            [    0.0, 0.0673,    0.0,-0.0022,    0.0,-0.2317]])

        FV_s = np.sqrt(self.R_fly)/(self.body_mass*np.sqrt(self.g))
        FW_s = 1.0/(self.body_mass*np.sqrt(self.g*self.R_fly))
        TV_s = 1.0/(self.body_mass*np.sqrt(self.g*self.R_fly))
        TW_s = 1.0/(self.body_mass*self.R_fly*np.sqrt(self.g*self.R_fly))

        # Aerodynamic and Inertial damping
        dFT_dx = dFT_dx_I+dFT_dx_A

        # Compute

        A = np.dot(self.M_inv,dFT_dx)
        B = np.dot(self.M_inv,self.dFT_du)

        print('A')
        print(A)
        print('')
        print('B')
        print(B)
        print('')

        # ensure muscle activation stays within bounds:
        u0Lc  = _u0L
        u1Lc  = _u1L
        u2Lc  = _u2L
        u3Lc  = _u3L
        u4Lc  = _u4L
        u5Lc  = _u5L
        u6Lc  = _u6L
        u7Lc  = _u7L
        u8Lc  = _u8L
        u9Lc  = _u9L
        u10Lc = _u10L
        u11Lc = _u11L

        u0Rc  = _u0R
        u1Rc  = _u1R
        u2Rc  = _u2R
        u3Rc  = _u3R
        u4Rc  = _u4R
        u5Rc  = _u5R
        u6Rc  = _u6R
        u7Rc  = _u7R
        u8Rc  = _u8R
        u9Rc  = _u9R
        u10Rc = _u10R
        u11Rc = _u11R

        # Microstepping:
        vx_upd = _vx
        vy_upd = _vy
        vz_upd = _vz
        wx_upd = _wx
        wy_upd = _wy
        wz_upd = _wz
        q0_upd = _q0
        qx_upd = _qx
        qy_upd = _qy
        qz_upd = _qz
        px_upd = _px
        py_upd = _py
        pz_upd = _pz

        N_i = 10
        d_i = 1.0/N_i

        for i in range(N_i):

            eom_x0  = -0.5*np.pi*self.R_fly*self.rho_air/self.body_mass*sign(vx_upd)*vx_upd**2+A[0,0]*vx_upd+A[0,1]*vy_upd+A[0,2]*vz_upd+A[0,3]*wx_upd+A[0,4]*wy_upd+A[0,5]*wz_upd-2.0*(qx_upd*qz_upd-qy_upd*q0_upd)
            eom_u0L = B[0,0]*u0Lc+B[0,1]*u1Lc+B[0,2]*u2Lc+B[0,3]*u3Lc+B[0,4]*u4Lc+B[0,5]*u5Lc+B[0,6]*u6Lc+B[0,7]*u7Lc+B[0,8]*u8Lc+B[0,9]*u9Lc+B[0,10]*u10Lc+B[0,11]*u11Lc
            eom_u0R = B[0,12]*u0Rc+B[0,13]*u1Rc+B[0,14]*u2Rc+B[0,15]*u3Rc+B[0,16]*u4Rc+B[0,17]*u5Rc+B[0,18]*u6Rc+B[0,19]*u7Rc+B[0,20]*u8Rc+B[0,21]*u9Rc+B[0,22]*u10Rc+B[0,23]*u11Rc
            vx_n  = vx_upd+(eom_x0+eom_u0L+eom_u0R)*d_i

            eom_x1  = -0.5*np.pi*self.R_fly*self.rho_air/self.body_mass*sign(vy_upd)*vy_upd**2+A[1,0]*vx_upd+A[1,1]*vy_upd+A[1,2]*vz_upd+A[1,3]*wx_upd+A[1,4]*wy_upd+A[1,5]*wz_upd-2.0*(qy_upd*qz_upd+qx_upd*q0_upd)
            eom_u1L = B[1,0]*u0Lc+B[1,1]*u1Lc+B[1,2]*u2Lc+B[1,3]*u3Lc+B[1,4]*u4Lc+B[1,5]*u5Lc+B[1,6]*u6Lc+B[1,7]*u7Lc+B[1,8]*u8Lc+B[1,9]*u9Lc+B[1,10]*u10Lc+B[1,11]*u11Lc
            eom_u1R = B[1,12]*u0Rc+B[1,13]*u1Rc+B[1,14]*u2Rc+B[1,15]*u3Rc+B[1,16]*u4Rc+B[1,17]*u5Rc+B[1,18]*u6Rc+B[1,19]*u7Rc+B[1,20]*u8Rc+B[1,21]*u9Rc+B[1,22]*u10Rc+B[1,23]*u11Rc
            vy_n  = vy_upd+(eom_x1+eom_u1L+eom_u1R)*d_i

            eom_x2  = -0.5*np.pi*self.R_fly*self.rho_air/self.body_mass*sign(vz_upd)*vz_upd**2+A[2,0]*vx_upd+A[2,1]*vy_upd+A[2,2]*vz_upd+A[2,3]*wx_upd+A[2,4]*wy_upd+A[2,5]*wz_upd-(2.0*(q0_upd**2+qz_upd**2)-1)
            eom_u2L = B[2,0]*u0Lc+B[2,1]*u1Lc+B[2,2]*u2Lc+B[2,3]*u3Lc+B[2,4]*u4Lc+B[2,5]*u5Lc+B[2,6]*u6Lc+B[2,7]*u7Lc+B[2,8]*u8Lc+B[2,9]*u9Lc+B[2,10]*u10Lc+B[2,11]*u11Lc
            eom_u2R = B[2,12]*u0Rc+B[2,13]*u1Rc+B[2,14]*u2Rc+B[2,15]*u3Rc+B[2,16]*u4Rc+B[2,17]*u5Rc+B[2,18]*u6Rc+B[2,19]*u7Rc+B[2,20]*u8Rc+B[2,21]*u9Rc+B[2,22]*u10Rc+B[2,23]*u11Rc
            vz_n  = vz_upd+(eom_x2+eom_u2L+eom_u2R)*d_i

            eom_x3  = A[3,0]*vx_upd+A[3,1]*vy_upd+A[3,2]*vz_upd+A[3,3]*wx_upd+A[3,4]*wy_upd+A[3,5]*wz_upd+(self.body_cg[2]/self.R_fly)*2.0*(qy_upd*qz_upd+qx_upd*q0_upd)
            eom_u3L = B[3,0]*u0Lc+B[3,1]*u1Lc+B[3,2]*u2Lc+B[3,3]*u3Lc+B[3,4]*u4Lc+B[3,5]*u5Lc+B[3,6]*u6Lc+B[3,7]*u7Lc+B[3,8]*u8Lc+B[3,9]*u9Lc+B[3,10]*u10Lc+B[3,11]*u11Lc
            eom_u3R = B[3,12]*u0Rc+B[3,13]*u1Rc+B[3,14]*u2Rc+B[3,15]*u3Rc+B[3,16]*u4Rc+B[3,17]*u5Rc+B[3,18]*u6Rc+B[3,19]*u7Rc+B[3,20]*u8Rc+B[3,21]*u9Rc+B[3,22]*u10Rc+B[3,23]*u11Rc
            wx_n  = wx_upd+(eom_x3+eom_u3L+eom_u3R)*d_i

            eom_x4  = A[4,0]*vx_upd+A[4,1]*vy_upd+A[4,2]*vz_upd+A[4,3]*wx_upd+A[4,4]*wy_upd+A[4,5]*wz_upd-(self.body_cg[2]/self.R_fly)*2.0*(qx_upd*qz_upd-qy_upd*q0_upd)+(self.body_cg[0]/self.R_fly)*(2.0*(q0_upd**2+qz_upd**2)-1)
            eom_u4L = B[4,0]*u0Lc+B[4,1]*u1Lc+B[4,2]*u2Lc+B[4,3]*u3Lc+B[4,4]*u4Lc+B[4,5]*u5Lc+B[4,6]*u6Lc+B[4,7]*u7Lc+B[4,8]*u8Lc+B[4,9]*u9Lc+B[4,10]*u10Lc+B[4,11]*u11Lc
            eom_u4R = B[4,12]*u0Rc+B[4,13]*u1Rc+B[4,14]*u2Rc+B[4,15]*u3Rc+B[4,16]*u4Rc+B[4,17]*u5Rc+B[4,18]*u6Rc+B[4,19]*u7Rc+B[4,20]*u8Rc+B[4,21]*u9Rc+B[4,22]*u10Rc+B[4,23]*u11Rc
            wy_n  = wy_upd+(eom_x4+eom_u4L+eom_u4R)*d_i

            eom_x5  = A[5,0]*vx_upd+A[5,1]*vy_upd+A[5,2]*vz_upd+A[5,3]*wx_upd+A[5,4]*wy_upd+A[5,5]*wz_upd-(self.body_cg[0]/self.R_fly)*2.0*(qy_upd*qz_upd+qx_upd*q0_upd)
            eom_u5L = B[5,0]*u0Lc+B[5,1]*u1Lc+B[5,2]*u2Lc+B[5,3]*u3Lc+B[5,4]*u4Lc+B[5,5]*u5Lc+B[5,6]*u6Lc+B[5,7]*u7Lc+B[5,8]*u8Lc+B[5,9]*u9Lc+B[5,10]*u10Lc+B[5,11]*u11Lc
            eom_u5R = B[5,12]*u0Rc+B[5,13]*u1Rc+B[5,14]*u2Rc+B[5,15]*u3Rc+B[5,16]*u4Rc+B[5,17]*u5Rc+B[5,18]*u6Rc+B[5,19]*u7Rc+B[5,20]*u8Rc+B[5,21]*u9Rc+B[5,22]*u10Rc+B[5,23]*u11Rc
            wz_n  = wz_upd+(eom_x5+eom_u5L+eom_u5R)*d_i

            q0_n = q0_upd+(-wx_upd*qx_upd-wy_upd*qy_upd-wz_upd*qz_upd)*0.5*d_i
            qx_n = qx_upd+(wx_upd*q0_upd+wz_upd*qy_upd-wy_upd*qz_upd)*0.5*d_i
            qy_n = qy_upd+(wy_upd*q0_upd-wz_upd*qx_upd+wx_upd*qz_upd)*0.5*d_i
            qz_n = qz_upd+(wz_upd*q0_upd+wy_upd*qx_upd-wx_upd*qy_upd)*0.5*d_i

            q_n_norm = sqrt(q0_n**2+qx_n**2+qy_n**2+qz_n**2)
            q0_n = if_else(q_n_norm<0.1,q0_n,q0_n/q_n_norm)
            qx_n = if_else(q_n_norm<0.1,qx_n,qx_n/q_n_norm)
            qy_n = if_else(q_n_norm<0.1,qy_n,qy_n/q_n_norm)
            qz_n = if_else(q_n_norm<0.1,qz_n,qz_n/q_n_norm)

            px_n = px_upd+((1.0-2*(qy_upd**2+qz_upd**2))*vx_upd+2*(qx_upd*qy_upd-qz_upd*q0_upd)*vy_upd+2*(qx_upd*qz_upd+qy_upd*q0_upd)*vz_upd)*d_i
            py_n = py_upd+(2*(qx_upd*qy_upd+qz_upd*q0_upd)*vx_upd+(1.0-2*(qx_upd**2+qz_upd**2))*vy_upd+2*(qy_upd*qz_upd-qx_upd*q0_upd)*vz_upd)*d_i
            pz_n = pz_upd+(2*(qx_upd*qz_upd-qy_upd*q0_upd)*vx_upd+2*(qy_upd*qz_upd+qx_upd*q0_upd)*vy_upd+(1.0-2*(qx_upd**2+qy_upd**2))*vz_upd)*d_i

            vx_upd = vx_n
            vy_upd = vy_n
            vz_upd = vz_n
            wx_upd = wx_n
            wy_upd = wy_n
            wz_upd = wz_n
            q0_upd = q0_n
            qx_upd = qx_n
            qy_upd = qy_n
            qz_upd = qz_n
            px_upd = px_n
            py_upd = py_n
            pz_upd = pz_n

        self.mpc_model.set_rhs('vx',vx_upd)
        self.mpc_model.set_rhs('vy',vy_upd)
        self.mpc_model.set_rhs('vz',vz_upd)
        self.mpc_model.set_rhs('wx',wx_upd)
        self.mpc_model.set_rhs('wy',wy_upd)
        self.mpc_model.set_rhs('wz',wz_upd)
        self.mpc_model.set_rhs('q0',q0_upd)
        self.mpc_model.set_rhs('qx',qx_upd)
        self.mpc_model.set_rhs('qy',qy_upd)
        self.mpc_model.set_rhs('qz',qz_upd)
        self.mpc_model.set_rhs('px',px_upd)
        self.mpc_model.set_rhs('py',py_upd)
        self.mpc_model.set_rhs('pz',pz_upd)

        # Build the model
        self.mpc_model.setup()

    def set_N_wbs(self,N_wbs_in):
        self.N_wbs = N_wbs_in

    def set_goal(self,x_goal_in):
        # non-dimensionalize
        x_g = np.copy(x_goal_in)
        x_g[0] = x_g[0]/np.sqrt(self.g*self.R_fly)
        x_g[1] = x_g[1]/np.sqrt(self.g*self.R_fly)
        x_g[2] = x_g[2]/np.sqrt(self.g*self.R_fly)
        x_g[3] = x_g[3]*np.sqrt(self.R_fly)/np.sqrt(self.g)
        x_g[4] = x_g[4]*np.sqrt(self.R_fly)/np.sqrt(self.g)
        x_g[5] = x_g[5]*np.sqrt(self.R_fly)/np.sqrt(self.g)
        x_g[10] = x_g[10]/self.R_fly
        x_g[11] = x_g[11]/self.R_fly
        x_g[12] = x_g[12]/self.R_fly
        self.x_goal = x_g

    def set_N_robust(self,N_robust_in):
        self.N_robust = N_robust_in

    def set_N_horizon(self,N_horizon_in):
        self.N_horizon = N_horizon_in

    def set_N_state(self,N_state_in):
        self.N_state = N_state_in

    def set_baseline_muscle_activity(self,m_mean_in):
        self.m_mean = m_mean_in

    def set_wingbeat_freq(self,f_in):
        self.f = f_in
        self.dt = 1.0/self.f

    def setup_controller(self):
        self.mpc_controller = do_mpc.controller.MPC(self.mpc_model)

        setup_mpc = {
            'n_robust': self.N_robust,
            'n_horizon': self.N_horizon, 
            't_step': self.dt,
            'state_discretization': 'disrete',
            'store_full_solution':True,
        }

        self.mpc_controller.set_param(**setup_mpc)

        # goal cost function
        m_term = 0.0
        l_term = 0.0
        for i in range(self.N_state):
            m_term += ((self.x_goal[i]-self.x[i])**2)*self.x_scaling[i]
            l_term += ((self.x_goal[i]-self.x[i])**2)*self.x_scaling[i]

        # set goal cost functions:
        self.mpc_controller.set_objective(mterm=m_term, lterm=l_term)

        self.mpc_controller.set_rterm(u0L=self.u_weights[0])
        self.mpc_controller.set_rterm(u1L=self.u_weights[1])
        self.mpc_controller.set_rterm(u2L=self.u_weights[2])
        self.mpc_controller.set_rterm(u3L=self.u_weights[3])
        self.mpc_controller.set_rterm(u4L=self.u_weights[4])
        self.mpc_controller.set_rterm(u5L=self.u_weights[5])
        self.mpc_controller.set_rterm(u6L=self.u_weights[6])
        self.mpc_controller.set_rterm(u7L=self.u_weights[7])
        self.mpc_controller.set_rterm(u8L=self.u_weights[8])
        self.mpc_controller.set_rterm(u9L=self.u_weights[9])
        self.mpc_controller.set_rterm(u10L=self.u_weights[10])
        self.mpc_controller.set_rterm(u11L=self.u_weights[11])

        self.mpc_controller.set_rterm(u0R=self.u_weights[0])
        self.mpc_controller.set_rterm(u1R=self.u_weights[1])
        self.mpc_controller.set_rterm(u2R=self.u_weights[2])
        self.mpc_controller.set_rterm(u3R=self.u_weights[3])
        self.mpc_controller.set_rterm(u4R=self.u_weights[4])
        self.mpc_controller.set_rterm(u5R=self.u_weights[5])
        self.mpc_controller.set_rterm(u6R=self.u_weights[6])
        self.mpc_controller.set_rterm(u7R=self.u_weights[7])
        self.mpc_controller.set_rterm(u8R=self.u_weights[8])
        self.mpc_controller.set_rterm(u9R=self.u_weights[9])
        self.mpc_controller.set_rterm(u10R=self.u_weights[10])
        self.mpc_controller.set_rterm(u11R=self.u_weights[11])

        # lower bounds of the states
        self.mpc_controller.bounds['lower','_x','vx'] = -100.0
        self.mpc_controller.bounds['lower','_x','vy'] = -100.0
        self.mpc_controller.bounds['lower','_x','vz'] = -100.0
        self.mpc_controller.bounds['lower','_x','wx'] = -100.0
        self.mpc_controller.bounds['lower','_x','wy'] = -100.0
        self.mpc_controller.bounds['lower','_x','wz'] = -100.0
        self.mpc_controller.bounds['lower','_x','q0'] = -1.0
        self.mpc_controller.bounds['lower','_x','qx'] = -1.0
        self.mpc_controller.bounds['lower','_x','qy'] = -1.0
        self.mpc_controller.bounds['lower','_x','qz'] = -1.0
        self.mpc_controller.bounds['lower','_x','px'] = -1000.0
        self.mpc_controller.bounds['lower','_x','py'] = -1000.0
        self.mpc_controller.bounds['lower','_x','pz'] = -1000.0

        # upper bounds of the states
        self.mpc_controller.bounds['upper','_x','vx'] = 100.0
        self.mpc_controller.bounds['upper','_x','vy'] = 100.0
        self.mpc_controller.bounds['upper','_x','vz'] = 100.0
        self.mpc_controller.bounds['upper','_x','wx'] = 100.0
        self.mpc_controller.bounds['upper','_x','wy'] = 100.0
        self.mpc_controller.bounds['upper','_x','wz'] = 100.0
        self.mpc_controller.bounds['upper','_x','q0'] = 1.0
        self.mpc_controller.bounds['upper','_x','qx'] = 1.0
        self.mpc_controller.bounds['upper','_x','qy'] = 1.0
        self.mpc_controller.bounds['upper','_x','qz'] = 1.0
        self.mpc_controller.bounds['upper','_x','px'] = 1000.0
        self.mpc_controller.bounds['upper','_x','py'] = 1000.0
        self.mpc_controller.bounds['upper','_x','pz'] = 1000.0

        # lower bounds of the input
        self.mpc_controller.bounds['lower','_u','u0L'] = -self.m_mean[0]
        self.mpc_controller.bounds['lower','_u','u1L'] = -self.m_mean[1]
        self.mpc_controller.bounds['lower','_u','u2L'] = -self.m_mean[2]
        self.mpc_controller.bounds['lower','_u','u3L'] = -self.m_mean[3]
        self.mpc_controller.bounds['lower','_u','u4L'] = -self.m_mean[4]
        self.mpc_controller.bounds['lower','_u','u5L'] = -self.m_mean[5]
        self.mpc_controller.bounds['lower','_u','u6L'] = -self.m_mean[6]
        self.mpc_controller.bounds['lower','_u','u7L'] = -self.m_mean[7]
        self.mpc_controller.bounds['lower','_u','u8L'] = -self.m_mean[8]
        self.mpc_controller.bounds['lower','_u','u9L'] = -self.m_mean[9]
        self.mpc_controller.bounds['lower','_u','u10L'] = -self.m_mean[10]
        self.mpc_controller.bounds['lower','_u','u11L'] = -self.m_mean[11]

        self.mpc_controller.bounds['lower','_u','u0R'] = -self.m_mean[0]
        self.mpc_controller.bounds['lower','_u','u1R'] = -self.m_mean[1]
        self.mpc_controller.bounds['lower','_u','u2R'] = -self.m_mean[2]
        self.mpc_controller.bounds['lower','_u','u3R'] = -self.m_mean[3]
        self.mpc_controller.bounds['lower','_u','u4R'] = -self.m_mean[4]
        self.mpc_controller.bounds['lower','_u','u5R'] = -self.m_mean[5]
        self.mpc_controller.bounds['lower','_u','u6R'] = -self.m_mean[6]
        self.mpc_controller.bounds['lower','_u','u7R'] = -self.m_mean[7]
        self.mpc_controller.bounds['lower','_u','u8R'] = -self.m_mean[8]
        self.mpc_controller.bounds['lower','_u','u9R'] = -self.m_mean[9]
        self.mpc_controller.bounds['lower','_u','u10R'] = -self.m_mean[10]
        self.mpc_controller.bounds['lower','_u','u11R'] = -self.m_mean[11]

        self.mpc_controller.bounds['upper','_u','u0L'] = (1.0-self.m_mean[0])
        self.mpc_controller.bounds['upper','_u','u1L'] = (1.0-self.m_mean[1])
        self.mpc_controller.bounds['upper','_u','u2L'] = (1.0-self.m_mean[2])
        self.mpc_controller.bounds['upper','_u','u3L'] = (1.0-self.m_mean[3])
        self.mpc_controller.bounds['upper','_u','u4L'] = (1.0-self.m_mean[4])
        self.mpc_controller.bounds['upper','_u','u5L'] = (1.0-self.m_mean[5])
        self.mpc_controller.bounds['upper','_u','u6L'] = (1.0-self.m_mean[6])
        self.mpc_controller.bounds['upper','_u','u7L'] = (1.0-self.m_mean[7])
        self.mpc_controller.bounds['upper','_u','u8L'] = (1.0-self.m_mean[8])
        self.mpc_controller.bounds['upper','_u','u9L'] = (1.0-self.m_mean[9])
        self.mpc_controller.bounds['upper','_u','u10L'] = (1.0-self.m_mean[10])
        self.mpc_controller.bounds['upper','_u','u11L'] = (1.0-self.m_mean[11])

        self.mpc_controller.bounds['upper','_u','u0R'] = (1.0-self.m_mean[0])
        self.mpc_controller.bounds['upper','_u','u1R'] = (1.0-self.m_mean[1])
        self.mpc_controller.bounds['upper','_u','u2R'] = (1.0-self.m_mean[2])
        self.mpc_controller.bounds['upper','_u','u3R'] = (1.0-self.m_mean[3])
        self.mpc_controller.bounds['upper','_u','u4R'] = (1.0-self.m_mean[4])
        self.mpc_controller.bounds['upper','_u','u5R'] = (1.0-self.m_mean[5])
        self.mpc_controller.bounds['upper','_u','u6R'] = (1.0-self.m_mean[6])
        self.mpc_controller.bounds['upper','_u','u7R'] = (1.0-self.m_mean[7])
        self.mpc_controller.bounds['upper','_u','u8R'] = (1.0-self.m_mean[8])
        self.mpc_controller.bounds['upper','_u','u9R'] = (1.0-self.m_mean[9])
        self.mpc_controller.bounds['upper','_u','u10R'] = (1.0-self.m_mean[10])
        self.mpc_controller.bounds['upper','_u','u11R'] = (1.0-self.m_mean[11])

        # enforce muscle plane:
        mc = self.muscle_coeffs

        #nl_tol = 0.0001
        nl_tol = 5.0
        max_viol = 0.1

        self.bias = np.zeros(12)

        self.mpc_controller.set_nl_cons('u0L_const',mc[0,0]*self.uL[0]+mc[0,1]*self.uL[1]+mc[0,2]*self.uL[2]+mc[0,3]*self.uL[3]+mc[0,4]*self.uL[4]+mc[0,5]*self.uL[5]+mc[0,6]*self.uL[6]+mc[0,7]*self.uL[7]+mc[0,8]*self.uL[8]+mc[0,9]*self.uL[9]+mc[0,10]*self.uL[10]+mc[0,11]*self.uL[11]-self.bias[0],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u1L_const',mc[1,0]*self.uL[0]+mc[1,1]*self.uL[1]+mc[1,2]*self.uL[2]+mc[1,3]*self.uL[3]+mc[1,4]*self.uL[4]+mc[1,5]*self.uL[5]+mc[1,6]*self.uL[6]+mc[1,7]*self.uL[7]+mc[1,8]*self.uL[8]+mc[1,9]*self.uL[9]+mc[1,10]*self.uL[10]+mc[1,11]*self.uL[11]-self.bias[1],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u2L_const',mc[2,0]*self.uL[0]+mc[2,1]*self.uL[1]+mc[2,2]*self.uL[2]+mc[2,3]*self.uL[3]+mc[2,4]*self.uL[4]+mc[2,5]*self.uL[5]+mc[2,6]*self.uL[6]+mc[2,7]*self.uL[7]+mc[2,8]*self.uL[8]+mc[2,9]*self.uL[9]+mc[2,10]*self.uL[10]+mc[2,11]*self.uL[11]-self.bias[2],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u3L_const',mc[3,0]*self.uL[0]+mc[3,1]*self.uL[1]+mc[3,2]*self.uL[2]+mc[3,3]*self.uL[3]+mc[3,4]*self.uL[4]+mc[3,5]*self.uL[5]+mc[3,6]*self.uL[6]+mc[3,7]*self.uL[7]+mc[3,8]*self.uL[8]+mc[3,9]*self.uL[9]+mc[3,10]*self.uL[10]+mc[3,11]*self.uL[11]-self.bias[3],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u4L_const',mc[4,0]*self.uL[0]+mc[4,1]*self.uL[1]+mc[4,2]*self.uL[2]+mc[4,3]*self.uL[3]+mc[4,4]*self.uL[4]+mc[4,5]*self.uL[5]+mc[4,6]*self.uL[6]+mc[4,7]*self.uL[7]+mc[4,8]*self.uL[8]+mc[4,9]*self.uL[9]+mc[4,10]*self.uL[10]+mc[4,11]*self.uL[11]-self.bias[4],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u5L_const',mc[5,0]*self.uL[0]+mc[5,1]*self.uL[1]+mc[5,2]*self.uL[2]+mc[5,3]*self.uL[3]+mc[5,4]*self.uL[4]+mc[5,5]*self.uL[5]+mc[5,6]*self.uL[6]+mc[5,7]*self.uL[7]+mc[5,8]*self.uL[8]+mc[5,9]*self.uL[9]+mc[5,10]*self.uL[10]+mc[5,11]*self.uL[11]-self.bias[5],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u6L_const',mc[6,0]*self.uL[0]+mc[6,1]*self.uL[1]+mc[6,2]*self.uL[2]+mc[6,3]*self.uL[3]+mc[6,4]*self.uL[4]+mc[6,5]*self.uL[5]+mc[6,6]*self.uL[6]+mc[6,7]*self.uL[7]+mc[6,8]*self.uL[8]+mc[6,9]*self.uL[9]+mc[6,10]*self.uL[10]+mc[6,11]*self.uL[11]-self.bias[6],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u7L_const',mc[7,0]*self.uL[0]+mc[7,1]*self.uL[1]+mc[7,2]*self.uL[2]+mc[7,3]*self.uL[3]+mc[7,4]*self.uL[4]+mc[7,5]*self.uL[5]+mc[7,6]*self.uL[6]+mc[7,7]*self.uL[7]+mc[7,8]*self.uL[8]+mc[7,9]*self.uL[9]+mc[7,10]*self.uL[10]+mc[7,11]*self.uL[11]-self.bias[7],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u8L_const',mc[8,0]*self.uL[0]+mc[8,1]*self.uL[1]+mc[8,2]*self.uL[2]+mc[8,3]*self.uL[3]+mc[8,4]*self.uL[4]+mc[8,5]*self.uL[5]+mc[8,6]*self.uL[6]+mc[8,7]*self.uL[7]+mc[8,8]*self.uL[8]+mc[8,9]*self.uL[9]+mc[8,10]*self.uL[10]+mc[8,11]*self.uL[11]-self.bias[8],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u9L_const',mc[9,0]*self.uL[0]+mc[9,1]*self.uL[1]+mc[9,2]*self.uL[2]+mc[9,3]*self.uL[3]+mc[9,4]*self.uL[4]+mc[9,5]*self.uL[5]+mc[9,6]*self.uL[6]+mc[9,7]*self.uL[7]+mc[9,8]*self.uL[8]+mc[9,9]*self.uL[9]+mc[9,10]*self.uL[10]+mc[9,11]*self.uL[11]-self.bias[9],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u10L_const',mc[10,0]*self.uL[0]+mc[10,1]*self.uL[1]+mc[10,2]*self.uL[2]+mc[10,3]*self.uL[3]+mc[10,4]*self.uL[4]+mc[10,5]*self.uL[5]+mc[10,6]*self.uL[6]+mc[10,7]*self.uL[7]+mc[10,8]*self.uL[8]+mc[10,9]*self.uL[9]+mc[10,10]*self.uL[10]+mc[10,11]*self.uL[11]-self.bias[10],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u11L_const',mc[11,0]*self.uL[0]+mc[11,1]*self.uL[1]+mc[11,2]*self.uL[2]+mc[11,3]*self.uL[3]+mc[11,4]*self.uL[4]+mc[11,5]*self.uL[5]+mc[11,6]*self.uL[6]+mc[11,7]*self.uL[7]+mc[11,8]*self.uL[8]+mc[11,9]*self.uL[9]+mc[11,10]*self.uL[10]+mc[11,11]*self.uL[11]-self.bias[11],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)

        self.mpc_controller.set_nl_cons('u0R_const',mc[0,0]*self.uR[0]+mc[0,1]*self.uR[1]+mc[0,2]*self.uR[2]+mc[0,3]*self.uR[3]+mc[0,4]*self.uR[4]+mc[0,5]*self.uR[5]+mc[0,6]*self.uR[6]+mc[0,7]*self.uR[7]+mc[0,8]*self.uR[8]+mc[0,9]*self.uR[9]+mc[0,10]*self.uR[10]+mc[0,11]*self.uR[11]-self.bias[0],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u1R_const',mc[1,0]*self.uR[0]+mc[1,1]*self.uR[1]+mc[1,2]*self.uR[2]+mc[1,3]*self.uR[3]+mc[1,4]*self.uR[4]+mc[1,5]*self.uR[5]+mc[1,6]*self.uR[6]+mc[1,7]*self.uR[7]+mc[1,8]*self.uR[8]+mc[1,9]*self.uR[9]+mc[1,10]*self.uR[10]+mc[1,11]*self.uR[11]-self.bias[1],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u2R_const',mc[2,0]*self.uR[0]+mc[2,1]*self.uR[1]+mc[2,2]*self.uR[2]+mc[2,3]*self.uR[3]+mc[2,4]*self.uR[4]+mc[2,5]*self.uR[5]+mc[2,6]*self.uR[6]+mc[2,7]*self.uR[7]+mc[2,8]*self.uR[8]+mc[2,9]*self.uR[9]+mc[2,10]*self.uR[10]+mc[2,11]*self.uR[11]-self.bias[2],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u3R_const',mc[3,0]*self.uR[0]+mc[3,1]*self.uR[1]+mc[3,2]*self.uR[2]+mc[3,3]*self.uR[3]+mc[3,4]*self.uR[4]+mc[3,5]*self.uR[5]+mc[3,6]*self.uR[6]+mc[3,7]*self.uR[7]+mc[3,8]*self.uR[8]+mc[3,9]*self.uR[9]+mc[3,10]*self.uR[10]+mc[3,11]*self.uR[11]-self.bias[3],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u4R_const',mc[4,0]*self.uR[0]+mc[4,1]*self.uR[1]+mc[4,2]*self.uR[2]+mc[4,3]*self.uR[3]+mc[4,4]*self.uR[4]+mc[4,5]*self.uR[5]+mc[4,6]*self.uR[6]+mc[4,7]*self.uR[7]+mc[4,8]*self.uR[8]+mc[4,9]*self.uR[9]+mc[4,10]*self.uR[10]+mc[4,11]*self.uR[11]-self.bias[4],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u5R_const',mc[5,0]*self.uR[0]+mc[5,1]*self.uR[1]+mc[5,2]*self.uR[2]+mc[5,3]*self.uR[3]+mc[5,4]*self.uR[4]+mc[5,5]*self.uR[5]+mc[5,6]*self.uR[6]+mc[5,7]*self.uR[7]+mc[5,8]*self.uR[8]+mc[5,9]*self.uR[9]+mc[5,10]*self.uR[10]+mc[5,11]*self.uR[11]-self.bias[5],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u6R_const',mc[6,0]*self.uR[0]+mc[6,1]*self.uR[1]+mc[6,2]*self.uR[2]+mc[6,3]*self.uR[3]+mc[6,4]*self.uR[4]+mc[6,5]*self.uR[5]+mc[6,6]*self.uR[6]+mc[6,7]*self.uR[7]+mc[6,8]*self.uR[8]+mc[6,9]*self.uR[9]+mc[6,10]*self.uR[10]+mc[6,11]*self.uR[11]-self.bias[6],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u7R_const',mc[7,0]*self.uR[0]+mc[7,1]*self.uR[1]+mc[7,2]*self.uR[2]+mc[7,3]*self.uR[3]+mc[7,4]*self.uR[4]+mc[7,5]*self.uR[5]+mc[7,6]*self.uR[6]+mc[7,7]*self.uR[7]+mc[7,8]*self.uR[8]+mc[7,9]*self.uR[9]+mc[7,10]*self.uR[10]+mc[7,11]*self.uR[11]-self.bias[7],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u8R_const',mc[8,0]*self.uR[0]+mc[8,1]*self.uR[1]+mc[8,2]*self.uR[2]+mc[8,3]*self.uR[3]+mc[8,4]*self.uR[4]+mc[8,5]*self.uR[5]+mc[8,6]*self.uR[6]+mc[8,7]*self.uR[7]+mc[8,8]*self.uR[8]+mc[8,9]*self.uR[9]+mc[8,10]*self.uR[10]+mc[8,11]*self.uR[11]-self.bias[8],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u9R_const',mc[9,0]*self.uR[0]+mc[9,1]*self.uR[1]+mc[9,2]*self.uR[2]+mc[9,3]*self.uR[3]+mc[9,4]*self.uR[4]+mc[9,5]*self.uR[5]+mc[9,6]*self.uR[6]+mc[9,7]*self.uR[7]+mc[9,8]*self.uR[8]+mc[9,9]*self.uR[9]+mc[9,10]*self.uR[10]+mc[9,11]*self.uR[11]-self.bias[9],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u10R_const',mc[10,0]*self.uR[0]+mc[10,1]*self.uR[1]+mc[10,2]*self.uR[2]+mc[10,3]*self.uR[3]+mc[10,4]*self.uR[4]+mc[10,5]*self.uR[5]+mc[10,6]*self.uR[6]+mc[10,7]*self.uR[7]+mc[10,8]*self.uR[8]+mc[10,9]*self.uR[9]+mc[10,10]*self.uR[10]+mc[10,11]*self.uR[11]-self.bias[10],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)
        self.mpc_controller.set_nl_cons('u11R_const',mc[11,0]*self.uR[0]+mc[11,1]*self.uR[1]+mc[11,2]*self.uR[2]+mc[11,3]*self.uR[3]+mc[11,4]*self.uR[4]+mc[11,5]*self.uR[5]+mc[11,6]*self.uR[6]+mc[11,7]*self.uR[7]+mc[11,8]*self.uR[8]+mc[11,9]*self.uR[9]+mc[11,10]*self.uR[10]+mc[11,11]*self.uR[11]-self.bias[11],soft_constraint=True,penalty_term_cons=nl_tol,maximum_violation=max_viol)

        self.mpc_controller.setup()

    def setup_estimator(self):
        self.estimator = do_mpc.estimator.StateFeedback(self.mpc_model)
        self.estimator.x0 = self.x_initial

    def setup_simulator(self):
        self.simulator = do_mpc.simulator.Simulator(self.mpc_model)
        self.simulator.set_param(t_step = self.dt)
        self.simulator.x0 = self.x_initial
        self.simulator.setup()

    def set_initial_state(self,x_start):
        # Seed
        np.random.seed(99)
        # non-dimensionalize
        x_i = np.copy(x_start)
        x_i[0] = x_i[0]/np.sqrt(self.g*self.R_fly)
        x_i[1] = x_i[1]/np.sqrt(self.g*self.R_fly)
        x_i[2] = x_i[2]/np.sqrt(self.g*self.R_fly)
        x_i[3] = x_i[3]*np.sqrt(self.R_fly)/np.sqrt(self.g)
        x_i[4] = x_i[4]*np.sqrt(self.R_fly)/np.sqrt(self.g)
        x_i[5] = x_i[5]*np.sqrt(self.R_fly)/np.sqrt(self.g)
        x_i[10] = x_i[10]/self.R_fly
        x_i[11] = x_i[11]/self.R_fly
        x_i[12] = x_i[12]/self.R_fly
        self.x_initial = x_i

    def set_scaling(self,scaling_in):
        self.x_scaling = scaling_in

    def set_input_weights(self,u_weights_in):
        self.u_weights = u_weights_in

    def run_mpc(self):

        self.mpc_controller.x0 = self.x_initial
        self.mpc_controller.set_initial_guess()

        x0 = np.copy(self.x_initial)
        for k in range(self.N_wbs):
            u0 =self.mpc_controller.make_step(x0)
            y_next = self.simulator.make_step(u0)
            x0 = self.estimator.make_step(y_next)

        print('mpc data _x')
        print(self.mpc_controller.data['_x'])

        print('mpc data _u')
        print(self.mpc_controller.data['_u'])

        print('mpc data _t')
        print(self.mpc_controller.data['_time'])

        self.t_result = np.zeros(self.N_wbs+1)
        self.t_result[0] = 0.0
        self.t_result[1:] = np.squeeze(self.mpc_controller.data['_time'])+self.dt

        self.x_result = np.zeros((self.N_state,self.N_wbs+1))
        self.x_result[:,0]  = np.squeeze(self.x_initial)
        self.x_result[:,1:] = np.transpose(self.mpc_controller.data['_x'])

        u_LR = self.mpc_controller.data['_u']

        self.uL_result = np.zeros((12,self.N_wbs+1))
        self.uL_result[:,0] = np.squeeze(self.m_mean)
        self.uL_result[:,1:] = np.matlib.repmat(self.m_mean,1,self.N_wbs)+np.transpose(u_LR[:,:12])

        self.uR_result = np.zeros((12,self.N_wbs+1))
        self.uR_result[:,0]  = np.squeeze(self.m_mean)
        self.uR_result[:,1:] = np.matlib.repmat(self.m_mean,1,self.N_wbs)+np.transpose(u_LR[:,12:])

    def plot_muscles(self):
        fig = plt.figure(figsize=(18,10))
        gs = fig.add_gridspec(4,2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[2,0])
        ax4 = fig.add_subplot(gs[3,0])
        ax5 = fig.add_subplot(gs[0,1])
        ax6 = fig.add_subplot(gs[1,1])
        ax7 = fig.add_subplot(gs[2,1])
        ax8 = fig.add_subplot(gs[3,1])

        ax1.plot(self.t_result*1000,self.uL_result[0,:],label='b1',color=self.c_muscle[0])
        ax1.plot(self.t_result*1000,self.uL_result[1,:],label='b2',color=self.c_muscle[1])
        ax1.plot(self.t_result*1000,self.uL_result[2,:],label='b3',color=self.c_muscle[2])
        ax2.plot(self.t_result*1000,self.uL_result[3,:],label='i1',color=self.c_muscle[3])
        ax2.plot(self.t_result*1000,self.uL_result[4,:],label='i2',color=self.c_muscle[4])
        ax3.plot(self.t_result*1000,self.uL_result[5,:],label='iii1',color=self.c_muscle[5])
        ax3.plot(self.t_result*1000,self.uL_result[6,:],label='iii2',color=self.c_muscle[6])
        ax3.plot(self.t_result*1000,self.uL_result[7,:],label='iii3',color=self.c_muscle[7])
        ax4.plot(self.t_result*1000,self.uL_result[8,:],label='iv1',color=self.c_muscle[8])
        ax4.plot(self.t_result*1000,self.uL_result[9,:],label='iv2',color=self.c_muscle[9])
        ax4.plot(self.t_result*1000,self.uL_result[10,:],label='iv3',color=self.c_muscle[10])
        ax4.plot(self.t_result*1000,self.uL_result[11,:],label='iv4',color=self.c_muscle[11])

        ax5.plot(self.t_result*1000,self.uR_result[0,:],label='b1',color=self.c_muscle[0])
        ax5.plot(self.t_result*1000,self.uR_result[1,:],label='b2',color=self.c_muscle[1])
        ax5.plot(self.t_result*1000,self.uR_result[2,:],label='b3',color=self.c_muscle[2])
        ax6.plot(self.t_result*1000,self.uR_result[3,:],label='i1',color=self.c_muscle[3])
        ax6.plot(self.t_result*1000,self.uR_result[4,:],label='i2',color=self.c_muscle[4])
        ax7.plot(self.t_result*1000,self.uR_result[5,:],label='iii1',color=self.c_muscle[5])
        ax7.plot(self.t_result*1000,self.uR_result[6,:],label='iii2',color=self.c_muscle[6])
        ax7.plot(self.t_result*1000,self.uR_result[7,:],label='iii3',color=self.c_muscle[7])
        ax8.plot(self.t_result*1000,self.uR_result[8,:],label='iv1',color=self.c_muscle[8])
        ax8.plot(self.t_result*1000,self.uR_result[9,:],label='iv2',color=self.c_muscle[9])
        ax8.plot(self.t_result*1000,self.uR_result[10,:],label='iv3',color=self.c_muscle[10])
        ax8.plot(self.t_result*1000,self.uR_result[11,:],label='iv4',color=self.c_muscle[11])

        ax1.set_ylim([-0.1,1.1])
        ax2.set_ylim([-0.1,1.1])
        ax3.set_ylim([-0.1,1.1])
        ax4.set_ylim([-0.1,1.1])
        ax5.set_ylim([-0.1,1.1])
        ax6.set_ylim([-0.1,1.1])
        ax7.set_ylim([-0.1,1.1])
        ax8.set_ylim([-0.1,1.1])

        ax5.legend()
        ax6.legend()
        ax7.legend()
        ax8.legend()

        adjust_spines(ax1,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(ax2,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(ax3,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(ax4,['left','bottom'],yticks=[0,1],xticks=[0,25,50],linewidth=0.8,spineColor='k')
        adjust_spines(ax5,[],linewidth=0.8,spineColor='k')
        adjust_spines(ax6,[],linewidth=0.8,spineColor='k')
        adjust_spines(ax7,[],linewidth=0.8,spineColor='k')
        adjust_spines(ax8,['bottom'],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        ax1.set_ylabel('b L')
        ax2.set_ylabel('i L')
        ax3.set_ylabel('iii L')
        ax4.set_ylabel('iv L')
        ax4.set_xlabel('t [ms]')
        ax8.set_xlabel('t [ms]')

    def plot_wing_motion(self):

        self.n_pts = 50

        # iterate over number of wingbeats and get the changes in wing motion
        m_L = np.zeros((self.N_wbs+1,9,13))
        m_R = np.zeros((self.N_wbs+1,9,13))
        m_stamp = np.zeros((9,13))
        for i in range(self.N_wbs+1):
            m_stamp[:,:12] = np.matlib.repmat(np.transpose(self.uL_result[:,i]),9,1)
            m_stamp[:,12] = np.ones(9)*0.5
            m_L[i,:,:] = m_stamp
            m_stamp[:,:12] = np.matlib.repmat(np.transpose(self.uR_result[:,i]),9,1)
            m_stamp[:,12] = np.ones(9)*0.5
            m_R[i,:,:] = m_stamp
        a_L = self.Wingkin_scale_inverse(self.predict(m_L))
        a_R = self.Wingkin_scale_inverse(self.predict(m_R))
        t_wb = np.linspace(0.0,self.dt,num=self.n_pts)
        X_theta = self.LegendrePolynomials(self.n_pts,self.N_pol_theta,1)
        X_eta = self.LegendrePolynomials(self.n_pts,self.N_pol_eta,1)
        X_phi = self.LegendrePolynomials(self.n_pts,self.N_pol_phi,1)
        X_xi = self.LegendrePolynomials(self.n_pts,self.N_pol_xi,1)

        self.A_theta_L     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        self.A_eta_L     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        self.A_phi_L     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        self.A_xi_L     = np.zeros((self.N_pol_xi,self.N_wbs+1))
        self.A_theta_R     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        self.A_eta_R     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        self.A_phi_R     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        self.A_xi_R     = np.zeros((self.N_pol_xi,self.N_wbs+1))

        fig = plt.figure(figsize=(18,10))
        gs = fig.add_gridspec(4,1)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[2,0])
        ax4 = fig.add_subplot(gs[3,0])

        for i in range(self.N_wbs+1):
            if i>0 and i<self.N_wbs:
                a_theta_L     = self.LegendreBC(a_L[i-1,0:20],a_L[i,0:20],a_L[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_L     = self.LegendreBC(a_L[i-1,20:44],a_L[i,20:44],a_L[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_L     = self.LegendreBC(a_L[i-1,44:60],a_L[i,44:60],a_L[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_L         = self.LegendreBC(a_L[i-1,60:80],a_L[i,60:80],a_L[i+1,60:80],self.N_pol_xi,self.N_const)
                a_theta_R     = self.LegendreBC(a_R[i-1,0:20],a_R[i,0:20],a_R[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_R     = self.LegendreBC(a_R[i-1,20:44],a_R[i,20:44],a_R[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_R     = self.LegendreBC(a_R[i-1,44:60],a_R[i,44:60],a_R[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_R         = self.LegendreBC(a_R[i-1,60:80],a_R[i,60:80],a_R[i+1,60:80],self.N_pol_xi,self.N_const)
            elif i==0:
                a_theta_L     = self.LegendreBC(a_L[i,0:20],a_L[i,0:20],a_L[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_L     = self.LegendreBC(a_L[i,20:44],a_L[i,20:44],a_L[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_L     = self.LegendreBC(a_L[i,44:60],a_L[i,44:60],a_L[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_L         = self.LegendreBC(a_L[i,60:80],a_L[i,60:80],a_L[i+1,60:80],self.N_pol_xi,self.N_const)
                a_theta_R     = self.LegendreBC(a_R[i,0:20],a_R[i,0:20],a_R[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_R     = self.LegendreBC(a_R[i,20:44],a_R[i,20:44],a_R[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_R     = self.LegendreBC(a_R[i,44:60],a_R[i,44:60],a_R[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_R         = self.LegendreBC(a_R[i,60:80],a_R[i,60:80],a_R[i+1,60:80],self.N_pol_xi,self.N_const)
            elif i==self.N_wbs:
                a_theta_L     = self.LegendreBC(a_L[i-1,0:20],a_L[i,0:20],a_L[i,0:20],self.N_pol_theta,self.N_const)
                a_eta_L     = self.LegendreBC(a_L[i-1,20:44],a_L[i,20:44],a_L[i,20:44],self.N_pol_eta,self.N_const)
                a_phi_L     = self.LegendreBC(a_L[i-1,44:60],a_L[i,44:60],a_L[i,44:60],self.N_pol_phi,self.N_const)
                a_xi_L         = self.LegendreBC(a_L[i-1,60:80],a_L[i,60:80],a_L[i,60:80],self.N_pol_xi,self.N_const)
                a_theta_R     = self.LegendreBC(a_R[i-1,0:20],a_R[i,0:20],a_R[i,0:20],self.N_pol_theta,self.N_const)
                a_eta_R     = self.LegendreBC(a_R[i-1,20:44],a_R[i,20:44],a_R[i,20:44],self.N_pol_eta,self.N_const)
                a_phi_R     = self.LegendreBC(a_R[i-1,44:60],a_R[i,44:60],a_R[i,44:60],self.N_pol_phi,self.N_const)
                a_xi_R         = self.LegendreBC(a_R[i-1,60:80],a_R[i,60:80],a_R[i,60:80],self.N_pol_xi,self.N_const)
            a_theta_avg     = self.LegendreBC(a_L[0,0:20],a_L[0,0:20],a_L[0,0:20],self.N_pol_theta,self.N_const)
            a_eta_avg         = self.LegendreBC(a_L[0,20:44],a_L[0,20:44],a_L[0,20:44],self.N_pol_eta,self.N_const)
            a_phi_avg         = self.LegendreBC(a_L[0,44:60],a_L[0,44:60],a_L[0,44:60],self.N_pol_phi,self.N_const)
            a_xi_avg        = self.LegendreBC(a_L[0,60:80],a_L[0,60:80],a_L[0,60:80],self.N_pol_xi,self.N_const)

            self.A_theta_L[:,i] = a_theta_L
            self.A_eta_L[:,i]     = a_eta_L
            self.A_phi_L[:,i]     = a_phi_L
            self.A_xi_L[:,i]     = a_xi_L
            self.A_theta_R[:,i] = a_theta_R
            self.A_eta_R[:,i]     = a_eta_R
            self.A_phi_R[:,i]     = a_phi_R
            self.A_xi_R[:,i]     = a_xi_R

            ax1.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_phi[:,:,0],a_phi_L),color='r')
            ax1.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_phi[:,:,0],a_phi_R),color='b')
            ax2.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_theta[:,:,0],a_theta_L),color='r')
            ax2.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_theta[:,:,0],a_theta_R),color='b')
            ax3.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_eta[:,:,0],a_eta_L),color='r')
            ax3.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_eta[:,:,0],a_eta_R),color='b')
            ax4.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_xi[:,:,0],a_xi_L),color='r')
            ax4.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_xi[:,:,0],a_xi_R),color='b')
            ax1.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_phi[:,:,0],a_phi_avg),color='k')
            ax2.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_theta[:,:,0],a_theta_avg),color='k')
            ax3.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_eta[:,:,0],a_eta_avg),color='k')
            ax4.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_xi[:,:,0],a_xi_avg),color='k')

        adjust_spines(ax1,['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(ax2,['left'],yticks=[-45,0,45],linewidth=0.8,spineColor='k')
        adjust_spines(ax3,['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(ax4,['left','bottom'],yticks=[-45,0,45],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        ax1.set_ylabel(r'$\phi$')
        ax2.set_ylabel(r'$\theta$')
        ax3.set_ylabel(r'$\eta$')
        ax4.set_ylabel(r'$\xi$')
        ax4.set_xlabel('t [ms]')

    def plot_body_dynamics(self):

        fig = plt.figure(figsize=(18,10))
        gs = fig.add_gridspec(4,1)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[2,0])
        ax4 = fig.add_subplot(gs[3,0])

        ax1.plot(self.t_result*1000,self.x_result[0,:]*np.sqrt(self.g*self.R_fly),label='vx',color='r')
        ax1.plot(self.t_result*1000,self.x_result[1,:]*np.sqrt(self.g*self.R_fly),label='vy',color='g')
        ax1.plot(self.t_result*1000,self.x_result[2,:]*np.sqrt(self.g*self.R_fly),label='vz',color='b')
        ax2.plot(self.t_result*1000,self.x_result[3,:]*np.sqrt(self.g)/np.sqrt(self.R_fly),label='wx',color='r')
        ax2.plot(self.t_result*1000,self.x_result[4,:]*np.sqrt(self.g)/np.sqrt(self.R_fly),label='wy',color='g')
        ax2.plot(self.t_result*1000,self.x_result[5,:]*np.sqrt(self.g)/np.sqrt(self.R_fly),label='wz',color='b')
        ax3.plot(self.t_result*1000,self.x_result[6,:],label='q0',color='m')
        ax3.plot(self.t_result*1000,self.x_result[7,:],label='qx',color='r')
        ax3.plot(self.t_result*1000,self.x_result[8,:],label='qy',color='g')
        ax3.plot(self.t_result*1000,self.x_result[9,:],label='qz',color='b')
        ax4.plot(self.t_result*1000,self.x_result[10,:]*self.R_fly,label='x',color='r')
        ax4.plot(self.t_result*1000,self.x_result[11,:]*self.R_fly,label='y',color='g')
        ax4.plot(self.t_result*1000,self.x_result[12,:]*self.R_fly,label='z',color='b')

        adjust_spines(ax1,['left'],linewidth=0.8,spineColor='k')
        adjust_spines(ax2,['left'],linewidth=0.8,spineColor='k')
        adjust_spines(ax3,['left'],linewidth=0.8,spineColor='k')
        adjust_spines(ax4,['left','bottom'],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        ax1.set_ylabel('v [mm/s]')
        ax1.legend()
        ax2.set_ylabel('w [rad/s]')
        ax2.legend()
        ax3.set_ylabel('q')
        ax3.legend()
        ax4.set_ylabel('p [mm]')
        ax4.legend()
        ax4.set_xlabel('t [ms]')

    def plot_oversight(self,man_name,save_loc):

        fig = plt.figure()
        fig.set_size_inches(20,16)
        gs = fig.add_gridspec(8,6)

        ax_bL = fig.add_subplot(gs[0,0])
        ax_iL = fig.add_subplot(gs[1,0])
        ax_iiiL = fig.add_subplot(gs[2,0])
        ax_ivL = fig.add_subplot(gs[3,0])

        ax_bR = fig.add_subplot(gs[0,1])
        ax_iR = fig.add_subplot(gs[1,1])
        ax_iiiR = fig.add_subplot(gs[2,1])
        ax_ivR = fig.add_subplot(gs[3,1])

        ax_v = fig.add_subplot(gs[0,2:])
        ax_w = fig.add_subplot(gs[1,2:])
        ax_q = fig.add_subplot(gs[2,2:])
        ax_p = fig.add_subplot(gs[3,2:])

        ax_phi      = fig.add_subplot(gs[4,:])
        ax_theta = fig.add_subplot(gs[5,:])
        ax_eta      = fig.add_subplot(gs[6,:])
        ax_xi      = fig.add_subplot(gs[7,:])

        # Left steering muscles:

        ax_bL.plot(self.t_result*1000,self.uL_result[0,:],label='b1',color=self.c_muscle[0])
        ax_bL.plot(self.t_result*1000,self.uL_result[1,:],label='b2',color=self.c_muscle[1])
        ax_bL.plot(self.t_result*1000,self.uL_result[2,:],label='b3',color=self.c_muscle[2])
        ax_iL.plot(self.t_result*1000,self.uL_result[3,:],label='i1',color=self.c_muscle[3])
        ax_iL.plot(self.t_result*1000,self.uL_result[4,:],label='i2',color=self.c_muscle[4])
        ax_iiiL.plot(self.t_result*1000,self.uL_result[5,:],label='iii1',color=self.c_muscle[5])
        ax_iiiL.plot(self.t_result*1000,self.uL_result[6,:],label='iii2',color=self.c_muscle[6])
        ax_iiiL.plot(self.t_result*1000,self.uL_result[7,:],label='iii3',color=self.c_muscle[7])
        ax_ivL.plot(self.t_result*1000,self.uL_result[8,:],label='iv1',color=self.c_muscle[8])
        ax_ivL.plot(self.t_result*1000,self.uL_result[9,:],label='iv2',color=self.c_muscle[9])
        ax_ivL.plot(self.t_result*1000,self.uL_result[10,:],label='iv3',color=self.c_muscle[10])
        ax_ivL.plot(self.t_result*1000,self.uL_result[11,:],label='iv4',color=self.c_muscle[11])

        # Right steering muscles:

        ax_bR.plot(self.t_result*1000,self.uR_result[0,:],label='b1',color=self.c_muscle[0])
        ax_bR.plot(self.t_result*1000,self.uR_result[1,:],label='b2',color=self.c_muscle[1])
        ax_bR.plot(self.t_result*1000,self.uR_result[2,:],label='b3',color=self.c_muscle[2])
        ax_iR.plot(self.t_result*1000,self.uR_result[3,:],label='i1',color=self.c_muscle[3])
        ax_iR.plot(self.t_result*1000,self.uR_result[4,:],label='i2',color=self.c_muscle[4])
        ax_iiiR.plot(self.t_result*1000,self.uR_result[5,:],label='iii1',color=self.c_muscle[5])
        ax_iiiR.plot(self.t_result*1000,self.uR_result[6,:],label='iii2',color=self.c_muscle[6])
        ax_iiiR.plot(self.t_result*1000,self.uR_result[7,:],label='iii3',color=self.c_muscle[7])
        ax_ivR.plot(self.t_result*1000,self.uR_result[8,:],label='iv1',color=self.c_muscle[8])
        ax_ivR.plot(self.t_result*1000,self.uR_result[9,:],label='iv2',color=self.c_muscle[9])
        ax_ivR.plot(self.t_result*1000,self.uR_result[10,:],label='iv3',color=self.c_muscle[10])
        ax_ivR.plot(self.t_result*1000,self.uR_result[11,:],label='iv4',color=self.c_muscle[11])

        # Body motion:

        ax_v.plot(self.t_result*1000,self.x_result[0,:]*np.sqrt(self.g*self.R_fly),label='vx',color='r')
        ax_v.plot(self.t_result*1000,self.x_result[1,:]*np.sqrt(self.g*self.R_fly),label='vy',color='g')
        ax_v.plot(self.t_result*1000,self.x_result[2,:]*np.sqrt(self.g*self.R_fly),label='vz',color='b')
        ax_w.plot(self.t_result*1000,self.x_result[3,:]*np.sqrt(self.g)/np.sqrt(self.R_fly),label='wx',color='r')
        ax_w.plot(self.t_result*1000,self.x_result[4,:]*np.sqrt(self.g)/np.sqrt(self.R_fly),label='wy',color='g')
        ax_w.plot(self.t_result*1000,self.x_result[5,:]*np.sqrt(self.g)/np.sqrt(self.R_fly),label='wz',color='b')
        ax_q.plot(self.t_result*1000,self.x_result[6,:],label='q0',color='m')
        ax_q.plot(self.t_result*1000,self.x_result[7,:],label='qx',color='r')
        ax_q.plot(self.t_result*1000,self.x_result[8,:],label='qy',color='g')
        ax_q.plot(self.t_result*1000,self.x_result[9,:],label='qz',color='b')
        ax_p.plot(self.t_result*1000,self.x_result[10,:]*self.R_fly,label='x',color='r')
        ax_p.plot(self.t_result*1000,self.x_result[11,:]*self.R_fly,label='y',color='g')
        ax_p.plot(self.t_result*1000,self.x_result[12,:]*self.R_fly,label='z',color='b')

        # Wing motion

        self.n_pts = 50

        # iterate over number of wingbeats and get the changes in wing motion
        m_L = np.zeros((self.N_wbs+1,9,13))
        m_R = np.zeros((self.N_wbs+1,9,13))
        m_stamp = np.zeros((9,13))
        for i in range(self.N_wbs+1):
            m_stamp[:,:12] = np.matlib.repmat(np.transpose(self.uL_result[:,i]),9,1)
            m_stamp[:,12] = np.ones(9)*0.5
            m_L[i,:,:] = m_stamp
            m_stamp[:,:12] = np.matlib.repmat(np.transpose(self.uR_result[:,i]),9,1)
            m_stamp[:,12] = np.ones(9)*0.5
            m_R[i,:,:] = m_stamp
        a_L = self.Wingkin_scale_inverse(self.predict(m_L))
        a_R = self.Wingkin_scale_inverse(self.predict(m_R))
        t_wb = np.linspace(0.0,self.dt,num=self.n_pts)
        X_theta = self.LegendrePolynomials(self.n_pts,self.N_pol_theta,1)
        X_eta = self.LegendrePolynomials(self.n_pts,self.N_pol_eta,1)
        X_phi = self.LegendrePolynomials(self.n_pts,self.N_pol_phi,1)
        X_xi = self.LegendrePolynomials(self.n_pts,self.N_pol_xi,1)

        self.A_theta_L     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        self.A_eta_L     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        self.A_phi_L     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        self.A_xi_L     = np.zeros((self.N_pol_xi,self.N_wbs+1))
        self.A_theta_R     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        self.A_eta_R     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        self.A_phi_R     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        self.A_xi_R     = np.zeros((self.N_pol_xi,self.N_wbs+1))

        for i in range(self.N_wbs+1):
            if i>0 and i<self.N_wbs:
                a_theta_L     = self.LegendreBC(a_L[i-1,0:20],a_L[i,0:20],a_L[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_L     = self.LegendreBC(a_L[i-1,20:44],a_L[i,20:44],a_L[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_L     = self.LegendreBC(a_L[i-1,44:60],a_L[i,44:60],a_L[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_L         = self.LegendreBC(a_L[i-1,60:80],a_L[i,60:80],a_L[i+1,60:80],self.N_pol_xi,self.N_const)
                a_theta_R     = self.LegendreBC(a_R[i-1,0:20],a_R[i,0:20],a_R[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_R     = self.LegendreBC(a_R[i-1,20:44],a_R[i,20:44],a_R[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_R     = self.LegendreBC(a_R[i-1,44:60],a_R[i,44:60],a_R[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_R         = self.LegendreBC(a_R[i-1,60:80],a_R[i,60:80],a_R[i+1,60:80],self.N_pol_xi,self.N_const)
            elif i==0:
                a_theta_L     = self.LegendreBC(a_L[i,0:20],a_L[i,0:20],a_L[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_L     = self.LegendreBC(a_L[i,20:44],a_L[i,20:44],a_L[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_L     = self.LegendreBC(a_L[i,44:60],a_L[i,44:60],a_L[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_L         = self.LegendreBC(a_L[i,60:80],a_L[i,60:80],a_L[i+1,60:80],self.N_pol_xi,self.N_const)
                a_theta_R     = self.LegendreBC(a_R[i,0:20],a_R[i,0:20],a_R[i+1,0:20],self.N_pol_theta,self.N_const)
                a_eta_R     = self.LegendreBC(a_R[i,20:44],a_R[i,20:44],a_R[i+1,20:44],self.N_pol_eta,self.N_const)
                a_phi_R     = self.LegendreBC(a_R[i,44:60],a_R[i,44:60],a_R[i+1,44:60],self.N_pol_phi,self.N_const)
                a_xi_R         = self.LegendreBC(a_R[i,60:80],a_R[i,60:80],a_R[i+1,60:80],self.N_pol_xi,self.N_const)
            elif i==self.N_wbs:
                a_theta_L     = self.LegendreBC(a_L[i-1,0:20],a_L[i,0:20],a_L[i,0:20],self.N_pol_theta,self.N_const)
                a_eta_L     = self.LegendreBC(a_L[i-1,20:44],a_L[i,20:44],a_L[i,20:44],self.N_pol_eta,self.N_const)
                a_phi_L     = self.LegendreBC(a_L[i-1,44:60],a_L[i,44:60],a_L[i,44:60],self.N_pol_phi,self.N_const)
                a_xi_L         = self.LegendreBC(a_L[i-1,60:80],a_L[i,60:80],a_L[i,60:80],self.N_pol_xi,self.N_const)
                a_theta_R     = self.LegendreBC(a_R[i-1,0:20],a_R[i,0:20],a_R[i,0:20],self.N_pol_theta,self.N_const)
                a_eta_R     = self.LegendreBC(a_R[i-1,20:44],a_R[i,20:44],a_R[i,20:44],self.N_pol_eta,self.N_const)
                a_phi_R     = self.LegendreBC(a_R[i-1,44:60],a_R[i,44:60],a_R[i,44:60],self.N_pol_phi,self.N_const)
                a_xi_R         = self.LegendreBC(a_R[i-1,60:80],a_R[i,60:80],a_R[i,60:80],self.N_pol_xi,self.N_const)
            a_theta_avg     = self.LegendreBC(a_L[0,0:20],a_L[0,0:20],a_L[0,0:20],self.N_pol_theta,self.N_const)
            a_eta_avg         = self.LegendreBC(a_L[0,20:44],a_L[0,20:44],a_L[0,20:44],self.N_pol_eta,self.N_const)
            a_phi_avg         = self.LegendreBC(a_L[0,44:60],a_L[0,44:60],a_L[0,44:60],self.N_pol_phi,self.N_const)
            a_xi_avg        = self.LegendreBC(a_L[0,60:80],a_L[0,60:80],a_L[0,60:80],self.N_pol_xi,self.N_const)

            self.A_theta_L[:,i] = a_theta_L
            self.A_eta_L[:,i]     = a_eta_L
            self.A_phi_L[:,i]     = a_phi_L
            self.A_xi_L[:,i]     = a_xi_L
            self.A_theta_R[:,i] = a_theta_R
            self.A_eta_R[:,i]     = a_eta_R
            self.A_phi_R[:,i]     = a_phi_R
            self.A_xi_R[:,i]     = a_xi_R

            ax_phi.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_phi[:,:,0],a_phi_L),color='r')
            ax_phi.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_phi[:,:,0],a_phi_R),color='b')
            ax_theta.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_theta[:,:,0],a_theta_L),color='r')
            ax_theta.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_theta[:,:,0],a_theta_R),color='b')
            ax_eta.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_eta[:,:,0],a_eta_L),color='r')
            ax_eta.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_eta[:,:,0],a_eta_R),color='b')
            ax_xi.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_xi[:,:,0],a_xi_L),color='r')
            ax_xi.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_xi[:,:,0],a_xi_R),color='b')
            ax_phi.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_phi[:,:,0],a_phi_avg),color='k')
            ax_theta.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_theta[:,:,0],a_theta_avg),color='k')
            ax_eta.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_eta[:,:,0],a_eta_avg),color='k')
            ax_xi.plot(t_wb*1000+self.dt*i*1000,(180/np.pi)*np.dot(X_xi[:,:,0],a_xi_avg),color='k')

        adjust_spines(ax_bL,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(ax_iL,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(ax_iiiL,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(ax_ivL,['left','bottom'],yticks=[0,1],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        adjust_spines(ax_bR,[],linewidth=0.8,spineColor='k')
        adjust_spines(ax_iR,[],linewidth=0.8,spineColor='k')
        adjust_spines(ax_iiiR,[],linewidth=0.8,spineColor='k')
        adjust_spines(ax_ivR,['bottom'],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        adjust_spines(ax_v,['left'],linewidth=0.8,spineColor='k')
        adjust_spines(ax_w,['left'],linewidth=0.8,spineColor='k')
        adjust_spines(ax_q,['left'],linewidth=0.8,spineColor='k')
        adjust_spines(ax_p,['left','bottom'],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        adjust_spines(ax_phi,['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(ax_theta,['left'],yticks=[-45,0,45],linewidth=0.8,spineColor='k')
        adjust_spines(ax_eta,['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(ax_xi,['left','bottom'],yticks=[-45,0,45],xticks=[0,25,50],linewidth=0.8,spineColor='k')

        os.chdir(save_loc)
        file_name = man_name+'.svg'
        fig.savefig(file_name, dpi=300)


    def spline_fit(self,x,y,N):
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.linspace(x[0],x[-1],num=N)
        ynew = interpolate.splev(xnew, tck, der=0)
        return ynew

    def take_image(self,img_width,img_height,p_scale,cam_pos,clip_range,view_up,save_loc,file_name):
        # Add axes:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(2.0,2.0,2.0)
        axes.SetXAxisLabelText('')
        axes.SetYAxisLabelText('')
        axes.SetZAxisLabelText('')
        axes.SetCylinderRadius(0.2)
        axes.SetConeRadius(0.2)
        #self.ren.AddActor(axes)
        # Set background:
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(img_width,img_height)
        # Get Camera:
        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)
        # Set view:
        camera.SetParallelScale(p_scale)
        camera.SetPosition(cam_pos[0],cam_pos[1],cam_pos[2])
        camera.SetClippingRange(clip_range[0],clip_range[1])
        camera.SetFocalPoint(0.0,0.0,0.0)
        camera.SetViewUp(view_up[0],view_up[1],view_up[2])
        camera.OrthogonalizeViewUp()
        # Render:
        self.renWin.Render()
        
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.renWin)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img_i = w2i.GetOutput()
        n_rows, n_cols, _ = img_i.GetDimensions()
        img_sc = img_i.GetPointData().GetScalars()
        np_img = vtk_to_numpy(img_sc)
        np_img = cv2.flip(np_img.reshape(n_cols,n_rows,3),0)
        cv_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
        os.chdir(save_loc)
        cv2.imwrite(file_name,cv_img)

    def set_wing_motion(self,a_theta_L,a_eta_L,a_phi_L,a_xi_L,a_theta_R,a_eta_R,a_phi_R,a_xi_R,n_pts):
        N_points = n_pts
        N_wingbeats = a_theta_L.shape[1]
        # Create tip traces:
        X_theta = self.LegendrePolynomials(N_points,self.N_pol_theta,1)
        X_eta     = self.LegendrePolynomials(N_points,self.N_pol_eta,1)
        X_phi     = self.LegendrePolynomials(N_points,self.N_pol_phi,1)
        X_xi     = self.LegendrePolynomials(N_points,self.N_pol_xi,1)

        self.state_mat_L = np.zeros((8,N_points*N_wingbeats))
        self.state_mat_R = np.zeros((8,N_points*N_wingbeats))
        x_L = np.zeros(8)
        x_R = np.zeros(8)
        for i in range(self.N_wbs):
            # Compute wing kinematic angles:
            theta_L     = np.dot(X_theta[:,:,0],a_theta_L[:,i])
            eta_L         = np.dot(X_eta[:,:,0],a_eta_L[:,i])
            phi_L         = np.dot(X_phi[:,:,0],a_phi_L[:,i])
            xi_L         = np.dot(X_xi[:,:,0],a_xi_L[:,i])
            theta_R     = np.dot(X_theta[:,:,0],a_theta_R[:,i])
            eta_R         = np.dot(X_eta[:,:,0],a_eta_R[:,i])
            phi_R         = np.dot(X_phi[:,:,0],a_phi_R[:,i])
            xi_R         = np.dot(X_xi[:,:,0],a_xi_R[:,i])
            for j in range(N_points):
                x_L[0] = phi_L[j]
                x_L[1] = theta_L[j]
                x_L[2] = eta_L[j]
                x_L[3] = -xi_L[j]
                x_L[4] = 0.0
                x_L[5] = 0.6
                x_L[6] = 0.0
                x_R[0] = phi_R[j]
                x_R[1] = theta_R[j]
                x_R[2] = eta_R[j]
                x_R[3] = -xi_R[j]
                x_R[4] = 0.0
                x_R[5] = -0.6
                x_R[6] = 0.0
                self.state_mat_L[:,i*N_points+j] = self.calculate_state_L(x_L)
                self.state_mat_R[:,i*N_points+j] = self.calculate_state_R(x_R)

    def calculate_state_L(self,x_in):
        # parameters
        phi = x_in[0]
        theta = x_in[1]
        eta = x_in[2]
        xi = x_in[3]
        root_x = x_in[4]
        root_y = x_in[5]
        root_z = x_in[6]
        # convert to quaternions:
        q_start = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        #q_start = np.array([1.0,0.0,0.0,0.0])
        q_phi = np.array([np.cos(-phi/2.0),np.sin(-phi/2.0),0.0,0.0])
        q_theta = np.array([np.cos(theta/2.0),0.0,0.0,np.sin(theta/2.0)])
        q_eta = np.array([np.cos(-eta/2.0),0.0,np.sin(-eta/2.0),0.0])
        q_L = self.q_mult(q_eta,self.q_mult(q_theta,self.q_mult(q_phi,q_start)))
        # state out:
        state_out = np.zeros(8)
        state_out[0] = q_L[0]
        state_out[1] = q_L[1]
        state_out[2] = q_L[2]
        state_out[3] = q_L[3]
        state_out[4] = root_x
        state_out[5] = root_y
        state_out[6] = root_z
        state_out[7] = xi
        return state_out

    def calculate_state_R(self,x_in):
        # parameters
        phi = x_in[0]
        theta = x_in[1]
        eta = x_in[2]
        xi = x_in[3]
        root_x = x_in[4]
        root_y = x_in[5]
        root_z = x_in[6]
        # convert to quaternions:
        q_start = np.array([np.cos(self.beta/2.0),0.0,np.sin(self.beta/2.0),0.0])
        #q_start = np.array([1.0,0.0,0.0,0.0])
        q_phi = np.array([np.cos(phi/2.0),np.sin(phi/2.0),0.0,0.0])
        q_theta = np.array([np.cos(-theta/2.0),0.0,0.0,np.sin(-theta/2.0)])
        q_eta = np.array([np.cos(-eta/2.0),0.0,np.sin(-eta/2.0),0.0])
        q_R = self.q_mult(q_eta,self.q_mult(q_theta,self.q_mult(q_phi,q_start)))
        # state out:
        state_out = np.zeros(8)
        state_out[0] = q_R[0]
        state_out[1] = q_R[1]
        state_out[2] = q_R[2]
        state_out[3] = q_R[3]
        state_out[4] = root_x
        state_out[5] = root_y
        state_out[6] = root_z
        state_out[7] = xi
        return state_out

    def make_video(self,fig_name,view_nr,snapshot_dir,snapshot_inter):
        s_thorax  = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0])
        s_head       = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.55,0.0,0.42])
        s_abdomen = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1])
        s_wing_L  = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        s_wing_R  = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        body_scale = [self.scale[0],self.scale[1],self.scale[2]]
        body_clr = [(0.7,0.7,0.7)]

        #self.SetModelState(s_head,s_thorax,s_abdomen,s_wing_L,s_wing_R)

        a_theta_L     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        a_eta_L     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        a_phi_L     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        a_xi_L         = np.zeros((self.N_pol_xi,self.N_wbs+1))
        a_theta_R     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        a_eta_R     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        a_phi_R     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        a_xi_R         = np.zeros((self.N_pol_xi,self.N_wbs+1))

        N = (self.N_wbs+1)*self.n_pts

        self.s_body = np.zeros((7,N))
        self.s_body[1,:] = self.spline_fit(self.t_result,self.x_result[7,:],N)
        self.s_body[2,:] = self.spline_fit(self.t_result,self.x_result[8,:],N)
        self.s_body[3,:] = self.spline_fit(self.t_result,self.x_result[9,:],N)
        self.s_body[4,:] = self.spline_fit(self.t_result,self.x_result[10,:],N)
        self.s_body[5,:] = self.spline_fit(self.t_result,self.x_result[11,:],N)
        self.s_body[6,:] = self.spline_fit(self.t_result,self.x_result[12,:],N)
        self.s_body[0,:] = 1.0-np.sqrt(np.power(self.s_body[1,:],2)+np.power(self.s_body[2,:],2)+np.power(self.s_body[3,:],2))

        self.s_body[1,:self.n_pts] = np.ones(self.n_pts)*self.x_result[7,0]
        self.s_body[2,:self.n_pts] = np.ones(self.n_pts)*self.x_result[8,0]
        self.s_body[3,:self.n_pts] = np.ones(self.n_pts)*self.x_result[9,0]
        self.s_body[4,:self.n_pts] = np.ones(self.n_pts)*self.x_result[10,0]
        self.s_body[5,:self.n_pts] = np.ones(self.n_pts)*self.x_result[11,0]
        self.s_body[6,:self.n_pts] = np.ones(self.n_pts)*self.x_result[12,0]
        self.s_body[0,:] = 1.0-np.sqrt(np.power(self.s_body[1,:],2)+np.power(self.s_body[2,:],2)+np.power(self.s_body[3,:],2))
        
        self.set_wing_motion(self.A_theta_L,self.A_eta_L,self.A_phi_L,self.A_xi_L,self.A_theta_R,self.A_eta_R,self.A_phi_R,self.A_xi_R,self.n_pts)
        video_dir  = snapshot_dir
        video_file = fig_name+'.avi'
        self.create_video(video_dir,video_file,view_nr,10,snapshot_dir,snapshot_inter,'moving',1000,800)

    def make_video2(self,fig_name,view_nr,snapshot_dir,snapshot_inter):
        s_thorax  = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0])
        s_head       = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.55,0.0,0.42])
        s_abdomen = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1])
        s_wing_L  = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        s_wing_R  = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        body_scale = [self.scale[0],self.scale[1],self.scale[2]]
        body_clr = [(0.7,0.7,0.7)]

        #self.SetModelState(s_head,s_thorax,s_abdomen,s_wing_L,s_wing_R)

        a_theta_L     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        a_eta_L     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        a_phi_L     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        a_xi_L         = np.zeros((self.N_pol_xi,self.N_wbs+1))
        a_theta_R     = np.zeros((self.N_pol_theta,self.N_wbs+1))
        a_eta_R     = np.zeros((self.N_pol_eta,self.N_wbs+1))
        a_phi_R     = np.zeros((self.N_pol_phi,self.N_wbs+1))
        a_xi_R         = np.zeros((self.N_pol_xi,self.N_wbs+1))

        N = (self.N_wbs+1)*self.n_pts

        self.s_body[1,:] = np.ones(N)*self.x_result[7,0]
        self.s_body[2,:] = np.ones(N)*self.x_result[8,0]
        self.s_body[3,:] = np.ones(N)*self.x_result[9,0]
        self.s_body[4,:] = np.ones(N)*self.x_result[10,0]
        self.s_body[5,:] = np.ones(N)*self.x_result[11,0]
        self.s_body[6,:] = np.ones(N)*self.x_result[12,0]
        self.s_body[0,:] = 1.0-np.sqrt(np.power(self.s_body[1,:],2)+np.power(self.s_body[2,:],2)+np.power(self.s_body[3,:],2))
        
        self.set_wing_motion(self.A_theta_L,self.A_eta_L,self.A_phi_L,self.A_xi_L,self.A_theta_R,self.A_eta_R,self.A_phi_R,self.A_xi_R,self.n_pts)
        video_dir  = snapshot_dir
        video_file = fig_name+'.avi'
        self.create_video(video_dir,video_file,view_nr,3,snapshot_dir,snapshot_inter,'stationary',1000,800)

    def create_video(self,video_dir,video_file,view_nr,scale_in,snapshot_dir,snapshot_inter,snapshot_name,img_w,img_h):
        width_img = img_w
        height_img = img_h

        s_t = np.array([np.cos((-35.0/180.0)*np.pi/2.0),0.0,np.sin((-35.0/180.0)*np.pi/2.0),0.0,0.0,0.0,0.0,1.0])
        s_h = np.array([np.cos((-5.0/180.0)*np.pi/2.0),0.0,np.sin((-5.0/180.0)*np.pi/2.0),0.0,0.6*0.9,0.0,0.42*0.9,1.0])
        s_a = np.array([np.cos((-70.0/180.0)*np.pi/2.0),0.0,np.sin((-70.0/180.0)*np.pi/2.0),0.0,0.0,0.0,-0.1*0.9,1.0])

        self.body_mdl.set_Color([(0.5,0.5,0.5)])
        self.wing_mdl_L.set_Color([(1.0,0.0,0.0),(1.0,0.0,0.0),0.3])
        self.wing_mdl_R.set_Color([(0.0,0.0,1.0),(0.0,0.0,1.0),0.3])

        self.wing_mdl_L.transform_wing(self.state_mat_L[:,0])
        self.wing_mdl_R.transform_wing(self.state_mat_R[:,0])
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(width_img,height_img)

        camera = self.ren.GetActiveCamera()
        camera.SetParallelProjection(True)

        if view_nr==0:
            # frontal 30 deg up
            camera.SetParallelScale(scale_in)
            camera.SetPosition(8.0, 12.0, 3.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==1:
            # rear view
            camera.SetParallelScale(scale_in)
            camera.SetPosition(-12, 0.0, 0.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==2:
            # top view
            camera.SetParallelScale(scale_in)
            camera.SetPosition(0.0, 0.0, 12.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(1.0,0.0,0.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==3:
            # side view
            camera.SetParallelScale(scale_in)
            camera.SetPosition(0.0, 12.0, 0.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()
        elif view_nr==4:
            # front view
            camera.SetParallelScale(scale_in)
            camera.SetPosition(12.0,0.0,0.0)
            camera.SetClippingRange(0.0,48.0)
            camera.SetFocalPoint(0.0,0.0,0.0)
            camera.SetViewUp(0.0,0.0,1.0)
            camera.OrthogonalizeViewUp()

        #self.iren.Initialize()
        self.renWin.Render()

        time.sleep(1.0)

        size = (width_img,height_img)

        os.chdir(video_dir)
        out = cv2.VideoWriter(video_file,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        N_steps = self.state_mat_L.shape[1]

        print(N_steps)

        s_head       = np.zeros(7)
        s_thorax  = np.zeros(7)
        s_abdomen = np.zeros(7)
        x_wing_L  = np.ones(4)
        s_wing_L  = np.zeros(8)
        x_wing_R  = np.ones(4)
        s_wing_R  = np.zeros(8)

        os.chdir(snapshot_dir)

        for i in range(N_steps-1):
            M_body       = self.quat_mat(self.s_body[:,i])
            q_head       = self.q_mult(self.s_body[:4,i],s_h[:4])
            p_head       = np.dot(M_body,s_h[4:])
            s_head[:4]       = q_head
            s_head[4:]       = p_head[:3]
            q_thorax       = self.q_mult(self.s_body[:4,i],s_t[:4])
            p_thorax       = np.dot(M_body,s_t[4:])
            s_thorax[:4]  = q_thorax
            s_thorax[4:]  = p_thorax[:3]
            q_abdomen       = self.q_mult(self.s_body[:4,i],s_a[:4])
            p_abdomen       = np.dot(M_body,s_a[4:])
            s_abdomen[:4] = q_abdomen
            s_abdomen[4:] = p_abdomen[:3]
            q_body           = self.s_body[:4,i]
            q_body[1]      = -q_body[1]
            q_body[2]      = -q_body[2]
            q_body[3]      = -q_body[3]
            q_wing_L       = self.q_mult(self.state_mat_L[:4,i],q_body)
            x_wing_L[:3]  = self.state_mat_L[4:7,i]
            p_wing_L       = np.dot(M_body,x_wing_L)
            s_wing_L[:4]  = q_wing_L
            s_wing_L[4:7] = p_wing_L[:3]
            s_wing_L[7]   = self.state_mat_L[7,i]
            q_wing_R       = self.q_mult(self.state_mat_R[:4,i],q_body)
            x_wing_R[:3]  = self.state_mat_R[4:7,i]
            p_wing_R       = np.dot(M_body,x_wing_R)
            s_wing_R[:4]  = q_wing_R
            s_wing_R[4:7] = p_wing_R[:3]
            s_wing_R[7]   = self.state_mat_R[7,i]
            self.body_mdl.transform_head(s_head)
            self.body_mdl.transform_thorax(s_thorax)
            self.body_mdl.transform_abdomen(s_abdomen)
            self.wing_mdl_L.transform_wing(s_wing_L)
            self.wing_mdl_R.transform_wing(s_wing_R)
            #time.sleep(0.01)
            self.renWin.Render()
            #Export a single frame
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(self.renWin)
            w2i.SetInputBufferTypeToRGB()
            w2i.ReadFrontBufferOff()
            w2i.Update()
            img_i = w2i.GetOutput()
            n_rows, n_cols, _ = img_i.GetDimensions()
            img_sc = img_i.GetPointData().GetScalars()
            np_img = vtk_to_numpy(img_sc)
            np_img = cv2.flip(np_img.reshape(n_cols,n_rows,3),0)
            cv_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
            out.write(cv_img)
            if i%snapshot_inter==0:
                s_name = snapshot_name+'_'+str(i)+'.jpg'
                cv2.imwrite(s_name,cv_img)
        time.sleep(1.0)
        out.release()
        # Reset body and wing models
        tip_start_L = np.array([-1.8,0.5,1.8])
        tip_start_R = np.array([-1.8,-0.5,1.8])
        print(tip_start_L)
        print(tip_start_R)
        self.wing_mdl_L.clear_root_tip_pts(np.zeros(3),tip_start_L[:3])
        self.wing_mdl_R.clear_root_tip_pts(np.zeros(3),tip_start_R[:3])
        self.renWin.Render()

    def create_plot_locations(self):
        # create workign_dir/plots/[plot type] subfolders

        plot_folders = [
                'hover_flight', 
                'forward', 
                'backward', 
                'sideward', 
                'downward', 
                'upward', 
                'forward_flight',
                'saccade_left',
                'saccade_right', 
                'loom_left', 
                'loom_right',
                'loom_front',
                ]

        for folder in plot_folders:
            folder_path = os.path.join(self.plots_dir, folder) 
            print(f'created:  {folder_path}')
            os.makedirs(folder_path,  exist_ok=True)

