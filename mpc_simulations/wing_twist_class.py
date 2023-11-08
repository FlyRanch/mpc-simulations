import sys
import vtk
import numpy as np
import os

# Construct a model of a wing:
class WingModel_L():

    def __init__(self):
        self.vein_clr = (0.01,0.01,0.01)
        self.mem_clr = (0.01,0.01,0.01)
        self.mem_opacity = 0.3
        self.L0_vein()
        self.L1_vein()
        self.L2_vein()
        self.L3_vein()
        self.L4_vein()
        self.L5_vein()
        self.C1_vein()
        self.C2_vein()
        self.C3_vein()
        self.A_vein()
        self.P_vein()
        self.membrane_0()
        self.membrane_1()
        self.membrane_2()
        self.membrane_3()
        self.wing_key_pts = np.array([
            [0.2313, 0.5711, 0.0, 1.0],
            [0.3253, 2.3205, 0.0, 1.0],
            [0.0, 2.6241, 0.0, 1.0],
            [-0.2386, 2.5591, 0.0, 1.0],
            [-0.7012, 1.5976, 0.0, 1.0],
            [-0.7880, 0.8892, 0.0, 1.0],
            [-0.4048, 1.2578, 0.0, 1.0],
            [-0.1952, 1.2868, 0.0, 1.0],
            [0.0072, 0.7157, 0.0, 1.0],
            [-0.0867, 0.0145, 0.0, 1.0]])
        self.root_pts_list = []
        self.root_pts_list.append(np.zeros(3))
        self.root_trace(self.root_pts_list,500)
        self.tip_pts_list = []
        self.tip_pts_list.append(np.zeros(3))
        self.tip_trace(self.tip_pts_list,500)
        # root axes
        self.set_root_axes()
        # booleans:
        self.root_trace_on = False
        self.tip_trace_on = True
        self.root_axes_on = False

    def L0_vein(self):
        # L0 vein:
        self.L0_vein_pts = vtk.vtkPoints()
        self.L0_vein_pts.SetDataTypeToFloat()
        self.L0_vein_pts.InsertPoint(0,(0.0578,0.0145,0.0))
        self.L0_vein_pts.InsertPoint(1,(0.1301,0.0940,0.0))
        self.L0_vein_pts.InsertPoint(2,(0.1735,0.1663,0.0))
        self.L0_vein_pts.InsertPoint(3,(0.2024,0.2386,0.0))
        self.L0_vein_pts.InsertPoint(4,(0.2241,0.3108,0.0))
        self.L0_vein_pts.InsertPoint(5,(0.2313,0.3831,0.0))
        self.L0_vein_pts.InsertPoint(6,(0.2458,0.4554,0.0))
        self.L0_vein_pts.InsertPoint(7,(0.2458,0.5277,0.0))
        self.L0_vein_pts.InsertPoint(8,(0.2313,0.5711,0.0))
        # Fit a spline to the points
        self.L0_spline = vtk.vtkParametricSpline()
        self.L0_spline.SetPoints(self.L0_vein_pts)
        self.L0_function_src = vtk.vtkParametricFunctionSource()
        self.L0_function_src.SetParametricFunction(self.L0_spline)
        self.L0_function_src.SetUResolution(self.L0_vein_pts.GetNumberOfPoints())
        self.L0_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L0_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L0_function_src.GetOutput().GetNumberOfPoints()
        self.L0_tube_radius.SetNumberOfTuples(N_spline)
        self.L0_tube_radius.SetName("TubeRadius")
        tMin = 0.02168
        tMax = 0.02602
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            #r = 1.0
            self.L0_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L0 = vtk.vtkPolyData()
        self.tubePolyData_L0 = self.L0_function_src.GetOutput()
        self.tubePolyData_L0.GetPointData().AddArray(self.L0_tube_radius)
        self.tubePolyData_L0.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L0 = vtk.vtkTubeFilter();
        self.tuber_L0.SetInputData(self.tubePolyData_L0);
        self.tuber_L0.SetNumberOfSides(6);
        self.tuber_L0.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.tuber_L0.Update()
        tuber_copy = self.tuber_L0.GetOutput()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L0)
        lineMapper.SetScalarRange(self.tubePolyData_L0.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L0.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L0.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L0 = vtk.vtkActor()
        self.tubeActor_L0.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L0.SetMapper(tubeMapper)

    def L1_vein(self):
        # L1 vein:
        self.L1_vein_pts = vtk.vtkPoints()
        self.L1_vein_pts.SetDataTypeToFloat()
        self.L1_vein_pts.InsertPoint(0,(0.0578,0.0145,0.0))
        self.L1_vein_pts.InsertPoint(1,(0.0361,0.0940,0.0))
        self.L1_vein_pts.InsertPoint(2,(0.0361,0.1663,0.0))
        self.L1_vein_pts.InsertPoint(3,(0.0506,0.2386,0.0))
        self.L1_vein_pts.InsertPoint(4,(0.0651,0.3109,0.0))
        self.L1_vein_pts.InsertPoint(5,(0.1012,0.3831,0.0))
        self.L1_vein_pts.InsertPoint(6,(0.1374,0.4554,0.0))
        self.L1_vein_pts.InsertPoint(7,(0.1952,0.5277,0.0))
        self.L1_vein_pts.InsertPoint(8,(0.2386,0.5639,0.0))
        self.L1_vein_pts.InsertPoint(9,(0.2458,0.6000,0.0))
        self.L1_vein_pts.InsertPoint(10,(0.2747,0.6723,0.0))
        self.L1_vein_pts.InsertPoint(11,(0.3036,0.7446,0.0))
        self.L1_vein_pts.InsertPoint(12,(0.3181,0.8169,0.0))
        self.L1_vein_pts.InsertPoint(13,(0.3398,0.8892,0.0))
        self.L1_vein_pts.InsertPoint(14,(0.3542,0.9615,0.0))
        self.L1_vein_pts.InsertPoint(15,(0.3687,1.0337,0.0))
        self.L1_vein_pts.InsertPoint(16,(0.3831,1.1060,0.0))
        self.L1_vein_pts.InsertPoint(17,(0.3904,1.1783,0.0))
        self.L1_vein_pts.InsertPoint(18,(0.3976,1.2506,0.0))
        self.L1_vein_pts.InsertPoint(19,(0.4048,1.3229,0.0))
        self.L1_vein_pts.InsertPoint(20,(0.4121,1.3952,0.0))
        self.L1_vein_pts.InsertPoint(21,(0.4193,1.4675,0.0))
        self.L1_vein_pts.InsertPoint(22,(0.4193,1.5400,0.0))
        self.L1_vein_pts.InsertPoint(23,(0.4265,1.6121,0.0))
        self.L1_vein_pts.InsertPoint(24,(0.4265,1.6844,0.0))
        self.L1_vein_pts.InsertPoint(25,(0.4337,1.7566,0.0))
        self.L1_vein_pts.InsertPoint(26,(0.4265,1.8289,0.0))
        self.L1_vein_pts.InsertPoint(27,(0.4193,1.9012,0.0))
        self.L1_vein_pts.InsertPoint(28,(0.4121,1.9735,0.0))
        self.L1_vein_pts.InsertPoint(29,(0.4048,2.0458,0.0))
        self.L1_vein_pts.InsertPoint(30,(0.3904,2.1181,0.0))
        self.L1_vein_pts.InsertPoint(31,(0.3759,2.1904,0.0))
        self.L1_vein_pts.InsertPoint(32,(0.3542,2.2627,0.0))
        self.L1_vein_pts.InsertPoint(33,(0.3253,2.3350,0.0))
        self.L1_vein_pts.InsertPoint(34,(0.2819,2.4073,0.0))
        self.L1_vein_pts.InsertPoint(35,(0.2241,2.4795,0.0))
        self.L1_vein_pts.InsertPoint(36,(0.1374,2.5518,0.0))
        self.L1_vein_pts.InsertPoint(37,(0.0867,2.5880,0.0))
        self.L1_vein_pts.InsertPoint(38,(0.0506,2.6097,0.0))
        self.L1_vein_pts.InsertPoint(39,(0.0,2.6241,0.0))
        # Fit a spline to the points
        self.L1_spline = vtk.vtkParametricSpline()
        self.L1_spline.SetPoints(self.L1_vein_pts)
        self.L1_function_src = vtk.vtkParametricFunctionSource()
        self.L1_function_src.SetParametricFunction(self.L1_spline)
        self.L1_function_src.SetUResolution(self.L1_vein_pts.GetNumberOfPoints())
        self.L1_function_src.Update()
        # Tube radius
        self.L1_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L1_function_src.GetOutput().GetNumberOfPoints()
        self.L1_tube_radius.SetNumberOfTuples(N_spline)
        self.L1_tube_radius.SetName("TubeRadius")
        tMin = 0.03614
        tMax = 0.01444
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L1_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L1 = vtk.vtkPolyData()
        self.tubePolyData_L1 = self.L1_function_src.GetOutput()
        self.tubePolyData_L1.GetPointData().AddArray(self.L1_tube_radius)
        self.tubePolyData_L1.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L1 = vtk.vtkTubeFilter();
        self.tuber_L1.SetInputData(self.tubePolyData_L1);
        self.tuber_L1.SetNumberOfSides(6);
        self.tuber_L1.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L1)
        lineMapper.SetScalarRange(self.tubePolyData_L1.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L1.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L1.GetScalarRange())
        # Line Actor
        lineActor_L1 = vtk.vtkActor()
        lineActor_L1.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L1 = vtk.vtkActor()
        self.tubeActor_L1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L1.SetMapper(tubeMapper)

    def L2_vein(self):
        # L2 vein:
        self.L2_vein_pts = vtk.vtkPoints()
        self.L2_vein_pts.SetDataTypeToFloat()
        self.L2_vein_pts.InsertPoint(0,(0.0361,0.0940,0.0))
        self.L2_vein_pts.InsertPoint(1,(-0.0145,0.1663,0.0))
        self.L2_vein_pts.InsertPoint(2,(-0.0145,0.2241,0.0))
        self.L2_vein_pts.InsertPoint(3,(0.0145,0.3109,0.0))
        self.L2_vein_pts.InsertPoint(4,(0.0506,0.3831,0.0))
        self.L2_vein_pts.InsertPoint(5,(0.0723,0.4554,0.0))
        self.L2_vein_pts.InsertPoint(6,(0.0795,0.5277,0.0))
        self.L2_vein_pts.InsertPoint(7,(0.0867,0.6000,0.0))
        self.L2_vein_pts.InsertPoint(8,(0.1084,0.6723,0.0))
        self.L2_vein_pts.InsertPoint(9,(0.1229,0.7446,0.0))
        self.L2_vein_pts.InsertPoint(10,(0.1374,0.8169,0.0))
        self.L2_vein_pts.InsertPoint(11,(0.1518,0.8892,0.0))
        self.L2_vein_pts.InsertPoint(12,(0.1735,0.9615,0.0))
        self.L2_vein_pts.InsertPoint(13,(0.1880,1.0337,0.0))
        self.L2_vein_pts.InsertPoint(14,(0.2096,1.1060,0.0))
        self.L2_vein_pts.InsertPoint(15,(0.2241,1.1783,0.0))
        self.L2_vein_pts.InsertPoint(16,(0.2386,1.2506,0.0))
        self.L2_vein_pts.InsertPoint(17,(0.2458,1.3229,0.0))
        self.L2_vein_pts.InsertPoint(18,(0.2602,1.3952,0.0))
        self.L2_vein_pts.InsertPoint(19,(0.2675,1.4675,0.0))
        self.L2_vein_pts.InsertPoint(20,(0.2747,1.5398,0.0))
        self.L2_vein_pts.InsertPoint(21,(0.2747,1.6121,0.0))
        self.L2_vein_pts.InsertPoint(22,(0.2747,1.6844,0.0))
        self.L2_vein_pts.InsertPoint(23,(0.2747,1.7566,0.0))
        self.L2_vein_pts.InsertPoint(24,(0.2819,1.8289,0.0))
        self.L2_vein_pts.InsertPoint(25,(0.2819,1.9012,0.0))
        self.L2_vein_pts.InsertPoint(26,(0.2819,1.9735,0.0))
        self.L2_vein_pts.InsertPoint(27,(0.2819,2.0458,0.0))
        self.L2_vein_pts.InsertPoint(28,(0.2892,2.1181,0.0))
        self.L2_vein_pts.InsertPoint(29,(0.2892,2.1904,0.0))
        self.L2_vein_pts.InsertPoint(30,(0.3036,2.2627,0.0))
        self.L2_vein_pts.InsertPoint(31,(0.3253,2.3205,0.0))
        # Fit a spline to the points
        self.L2_spline = vtk.vtkParametricSpline()
        self.L2_spline.SetPoints(self.L2_vein_pts)
        self.L2_function_src = vtk.vtkParametricFunctionSource()
        self.L2_function_src.SetParametricFunction(self.L2_spline)
        self.L2_function_src.SetUResolution(self.L2_vein_pts.GetNumberOfPoints())
        self.L2_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L2_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L2_function_src.GetOutput().GetNumberOfPoints()
        self.L2_tube_radius.SetNumberOfTuples(N_spline)
        self.L2_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.01156
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L2_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L2 = vtk.vtkPolyData()
        self.tubePolyData_L2 = self.L2_function_src.GetOutput()
        self.tubePolyData_L2.GetPointData().AddArray(self.L2_tube_radius)
        self.tubePolyData_L2.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L2 = vtk.vtkTubeFilter();
        self.tuber_L2.SetInputData(self.tubePolyData_L2);
        self.tuber_L2.SetNumberOfSides(6);
        self.tuber_L2.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L2)
        lineMapper.SetScalarRange(self.tubePolyData_L2.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L2.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetScalarRange(self.tubePolyData_L2.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L2 = vtk.vtkActor()
        self.tubeActor_L2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L2.SetMapper(tubeMapper)

    def L3_vein(self):
        # L3 vein:
        self.L3_vein_pts = vtk.vtkPoints()
        self.L3_vein_pts.SetDataTypeToFloat()
        self.L3_vein_pts.InsertPoint(0,(-0.0145,0.2241,0.0))
        self.L3_vein_pts.InsertPoint(1,(0.0,0.3108,0.0))
        self.L3_vein_pts.InsertPoint(2,(0.0145,0.3831,0.0))
        self.L3_vein_pts.InsertPoint(3,(0.0145,0.4554,0.0))
        self.L3_vein_pts.InsertPoint(4,(0.0072,0.5277,0.0))
        self.L3_vein_pts.InsertPoint(5,(0.0072,0.6000,0.0))
        self.L3_vein_pts.InsertPoint(6,(0.0,0.6723,0.0))
        self.L3_vein_pts.InsertPoint(7,(0.0072,0.7157,0.0))
        self.L3_vein_pts.InsertPoint(8,(0.0145,0.7446,0.0))
        self.L3_vein_pts.InsertPoint(9,(0.0217,0.8169,0.0))
        self.L3_vein_pts.InsertPoint(10,(0.0361,0.8892,0.0))
        self.L3_vein_pts.InsertPoint(11,(0.0434,0.9615,0.0))
        self.L3_vein_pts.InsertPoint(12,(0.0506,1.0337,0.0))
        self.L3_vein_pts.InsertPoint(13,(0.0578,1.1060,0.0))
        self.L3_vein_pts.InsertPoint(14,(0.0578,1.1783,0.0))
        self.L3_vein_pts.InsertPoint(15,(0.0578,1.2506,0.0))
        self.L3_vein_pts.InsertPoint(16,(0.0578,1.3229,0.0))
        self.L3_vein_pts.InsertPoint(17,(0.0578,1.3952,0.0))
        self.L3_vein_pts.InsertPoint(18,(0.0651,1.4675,0.0))
        self.L3_vein_pts.InsertPoint(19,(0.0651,1.5398,0.0))
        self.L3_vein_pts.InsertPoint(20,(0.0723,1.6121,0.0))
        self.L3_vein_pts.InsertPoint(21,(0.0723,1.6844,0.0))
        self.L3_vein_pts.InsertPoint(22,(0.0723,1.7566,0.0))
        self.L3_vein_pts.InsertPoint(23,(0.0723,1.8289,0.0))
        self.L3_vein_pts.InsertPoint(24,(0.0651,1.9012,0.0))
        self.L3_vein_pts.InsertPoint(25,(0.0651,1.9735,0.0))
        self.L3_vein_pts.InsertPoint(26,(0.0578,2.0458,0.0))
        self.L3_vein_pts.InsertPoint(27,(0.0578,2.1181,0.0))
        self.L3_vein_pts.InsertPoint(28,(0.0506,2.1904,0.0))
        self.L3_vein_pts.InsertPoint(29,(0.0506,2.2627,0.0))
        self.L3_vein_pts.InsertPoint(30,(0.0361,2.3350,0.0))
        self.L3_vein_pts.InsertPoint(31,(0.0217,2.4073,0.0))
        self.L3_vein_pts.InsertPoint(32,(0.0145,2.4795,0.0))
        self.L3_vein_pts.InsertPoint(33,(0.0072,2.5518,0.0))
        self.L3_vein_pts.InsertPoint(34,(0.0,2.6241,0.0))
        # Fit a spline to the points
        self.L3_spline = vtk.vtkParametricSpline()
        self.L3_spline.SetPoints(self.L3_vein_pts)
        self.L3_function_src = vtk.vtkParametricFunctionSource()
        self.L3_function_src.SetParametricFunction(self.L3_spline)
        self.L3_function_src.SetUResolution(self.L3_vein_pts.GetNumberOfPoints())
        self.L3_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L3_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L3_function_src.GetOutput().GetNumberOfPoints()
        self.L3_tube_radius.SetNumberOfTuples(N_spline)
        self.L3_tube_radius.SetName("TubeRadius")
        tMin = 0.01302
        tMax = 0.01302
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L3_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L3 = vtk.vtkPolyData()
        self.tubePolyData_L3 = self.L3_function_src.GetOutput()
        self.tubePolyData_L3.GetPointData().AddArray(self.L3_tube_radius)
        self.tubePolyData_L3.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L3 = vtk.vtkTubeFilter();
        self.tuber_L3.SetInputData(self.tubePolyData_L3);
        self.tuber_L3.SetNumberOfSides(6);
        self.tuber_L3.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L3)
        lineMapper.SetScalarRange(self.tubePolyData_L3.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L3.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L3.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L3 = vtk.vtkActor()
        self.tubeActor_L3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L3.SetMapper(tubeMapper)

    def L4_vein(self):
        # L4 vein:
        self.L4_vein_pts = vtk.vtkPoints()
        self.L4_vein_pts.SetDataTypeToFloat()
        self.L4_vein_pts.InsertPoint(0,(-0.0867,0.0145,0.0))
        self.L4_vein_pts.InsertPoint(1,(-0.0651,0.0940,0.0))
        self.L4_vein_pts.InsertPoint(2,(-0.0506,0.1663,0.0))
        self.L4_vein_pts.InsertPoint(3,(-0.0723,0.2386,0.0))
        self.L4_vein_pts.InsertPoint(4,(-0.0795,0.3108,0.0))
        self.L4_vein_pts.InsertPoint(5,(-0.0795,0.3831,0.0))
        self.L4_vein_pts.InsertPoint(6,(-0.0867,0.4554,0.0))
        self.L4_vein_pts.InsertPoint(7,(-0.0867,0.5277,0.0))
        self.L4_vein_pts.InsertPoint(8,(-0.0940,0.6000,0.0))
        self.L4_vein_pts.InsertPoint(9,(-0.0867,0.6723,0.0))
        self.L4_vein_pts.InsertPoint(10,(-0.0795,0.6940,0.0))
        self.L4_vein_pts.InsertPoint(11,(-0.1012,0.7446,0.0))
        self.L4_vein_pts.InsertPoint(12,(-0.1084,0.8169,0.0))
        self.L4_vein_pts.InsertPoint(13,(-0.1301,0.8892,0.0))
        self.L4_vein_pts.InsertPoint(14,(-0.1446,0.9615,0.0))
        self.L4_vein_pts.InsertPoint(15,(-0.1518,1.0337,0.0))
        self.L4_vein_pts.InsertPoint(16,(-0.1663,1.1060,0.0))
        self.L4_vein_pts.InsertPoint(17,(-0.1807,1.1783,0.0))
        self.L4_vein_pts.InsertPoint(18,(-0.1880,1.2506,0.0))
        self.L4_vein_pts.InsertPoint(19,(-0.2024,1.2868,0.0))
        self.L4_vein_pts.InsertPoint(20,(-0.1952,1.3229,0.0))
        self.L4_vein_pts.InsertPoint(21,(-0.1880,1.3952,0.0))
        self.L4_vein_pts.InsertPoint(22,(-0.1880,1.4675,0.0))
        self.L4_vein_pts.InsertPoint(23,(-0.1880,1.5398,0.0))
        self.L4_vein_pts.InsertPoint(24,(-0.1880,1.6121,0.0))
        self.L4_vein_pts.InsertPoint(25,(-0.1880,1.6844,0.0))
        self.L4_vein_pts.InsertPoint(26,(-0.1880,1.7566,0.0))
        self.L4_vein_pts.InsertPoint(27,(-0.1880,1.8289,0.0))
        self.L4_vein_pts.InsertPoint(28,(-0.1880,1.9012,0.0))
        self.L4_vein_pts.InsertPoint(29,(-0.1952,1.9735,0.0))
        self.L4_vein_pts.InsertPoint(30,(-0.1952,2.0458,0.0))
        self.L4_vein_pts.InsertPoint(31,(-0.1952,2.1181,0.0))
        self.L4_vein_pts.InsertPoint(32,(-0.1952,2.1904,0.0))
        self.L4_vein_pts.InsertPoint(33,(-0.2024,2.2627,0.0))
        self.L4_vein_pts.InsertPoint(34,(-0.2024,2.3350,0.0))
        self.L4_vein_pts.InsertPoint(35,(-0.2096,2.4073,0.0))
        self.L4_vein_pts.InsertPoint(36,(-0.2169,2.4795,0.0))
        self.L4_vein_pts.InsertPoint(37,(-0.2386,2.5591,0.0))
        # Fit a spline to the points
        self.L4_spline = vtk.vtkParametricSpline()
        self.L4_spline.SetPoints(self.L4_vein_pts)
        self.L4_function_src = vtk.vtkParametricFunctionSource()
        self.L4_function_src.SetParametricFunction(self.L4_spline)
        self.L4_function_src.SetUResolution(self.L4_vein_pts.GetNumberOfPoints())
        self.L4_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L4_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L4_function_src.GetOutput().GetNumberOfPoints()
        self.L4_tube_radius.SetNumberOfTuples(N_spline)
        self.L4_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.00867
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L4_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L4 = vtk.vtkPolyData()
        self.tubePolyData_L4 = self.L4_function_src.GetOutput()
        self.tubePolyData_L4.GetPointData().AddArray(self.L4_tube_radius)
        self.tubePolyData_L4.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L4 = vtk.vtkTubeFilter();
        self.tuber_L4.SetInputData(self.tubePolyData_L4);
        self.tuber_L4.SetNumberOfSides(6);
        self.tuber_L4.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L4)
        lineMapper.SetScalarRange(self.tubePolyData_L4.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L4.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L4.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L4 = vtk.vtkActor()
        self.tubeActor_L4.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L4.SetMapper(tubeMapper)

    def L5_vein(self):
        # L5 vein:
        self.L5_vein_pts = vtk.vtkPoints()
        self.L5_vein_pts.SetDataTypeToFloat()
        self.L5_vein_pts.InsertPoint(0,(-0.0867,0.0145,0.0))
        self.L5_vein_pts.InsertPoint(1,(-0.1229,0.0940,0.0))
        self.L5_vein_pts.InsertPoint(2,(-0.1446,0.1663,0.0))
        self.L5_vein_pts.InsertPoint(3,(-0.1663,0.2386,0.0))
        self.L5_vein_pts.InsertPoint(4,(-0.1807,0.3108,0.0))
        self.L5_vein_pts.InsertPoint(5,(-0.1952,0.3831,0.0))
        self.L5_vein_pts.InsertPoint(6,(-0.2024,0.4554,0.0))
        self.L5_vein_pts.InsertPoint(7,(-0.2241,0.5277,0.0))
        self.L5_vein_pts.InsertPoint(8,(-0.2386,0.6000,0.0))
        self.L5_vein_pts.InsertPoint(9,(-0.2602,0.6723,0.0))
        self.L5_vein_pts.InsertPoint(10,(-0.2747,0.7446,0.0))
        self.L5_vein_pts.InsertPoint(11,(-0.2892,0.8169,0.0))
        self.L5_vein_pts.InsertPoint(12,(-0.3108,0.8892,0.0))
        self.L5_vein_pts.InsertPoint(13,(-0.3253,0.9615,0.0))
        self.L5_vein_pts.InsertPoint(14,(-0.3470,1.0337,0.0))
        self.L5_vein_pts.InsertPoint(15,(-0.3615,1.1060,0.0))
        self.L5_vein_pts.InsertPoint(16,(-0.3904,1.1783,0.0))
        self.L5_vein_pts.InsertPoint(17,(-0.4048,1.2506,0.0))
        self.L5_vein_pts.InsertPoint(18,(-0.4410,1.3229,0.0))
        self.L5_vein_pts.InsertPoint(19,(-0.4988,1.4313,0.0))
        self.L5_vein_pts.InsertPoint(20,(-0.5494,1.4675,0.0))
        self.L5_vein_pts.InsertPoint(21,(-0.6217,1.5398,0.0))
        self.L5_vein_pts.InsertPoint(22,(-0.7012,1.5976,0.0))
        # Fit a spline to the points
        self.L5_spline = vtk.vtkParametricSpline()
        self.L5_spline.SetPoints(self.L5_vein_pts)
        self.L5_function_src = vtk.vtkParametricFunctionSource()
        self.L5_function_src.SetParametricFunction(self.L5_spline)
        self.L5_function_src.SetUResolution(self.L5_vein_pts.GetNumberOfPoints())
        self.L5_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L5_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L5_function_src.GetOutput().GetNumberOfPoints()
        self.L5_tube_radius.SetNumberOfTuples(N_spline)
        self.L5_tube_radius.SetName("TubeRadius")
        tMin = 0.01734
        tMax = 0.00725
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L5_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L5 = vtk.vtkPolyData()
        self.tubePolyData_L5 = self.L5_function_src.GetOutput()
        self.tubePolyData_L5.GetPointData().AddArray(self.L5_tube_radius)
        self.tubePolyData_L5.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L5 = vtk.vtkTubeFilter();
        self.tuber_L5.SetInputData(self.tubePolyData_L5);
        self.tuber_L5.SetNumberOfSides(6);
        self.tuber_L5.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L5)
        lineMapper.SetScalarRange(self.tubePolyData_L5.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L5.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L5.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L5 = vtk.vtkActor()
        self.tubeActor_L5.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L5.SetMapper(tubeMapper)

    def C1_vein(self):
        # C1 vein:
        self.C1_vein_pts = vtk.vtkPoints()
        self.C1_vein_pts.SetDataTypeToFloat()
        self.C1_vein_pts.InsertPoint(0,(0.0,2.6241,0.0))
        self.C1_vein_pts.InsertPoint(1,(-0.0578,2.6241,0.0))
        self.C1_vein_pts.InsertPoint(2,(-0.1374,2.6097,0.0))
        self.C1_vein_pts.InsertPoint(3,(-0.1880,2.5880,0.0))
        self.C1_vein_pts.InsertPoint(4,(-0.2386,2.5518,0.0))
        # Fit a spline to the points
        self.C1_spline = vtk.vtkParametricSpline()
        self.C1_spline.SetPoints(self.C1_vein_pts)
        self.C1_function_src = vtk.vtkParametricFunctionSource()
        self.C1_function_src.SetParametricFunction(self.C1_spline)
        self.C1_function_src.SetUResolution(self.C1_vein_pts.GetNumberOfPoints())
        self.C1_function_src.Update()
        # Tube radius
        self.C1_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.C1_function_src.GetOutput().GetNumberOfPoints()
        self.C1_tube_radius.SetNumberOfTuples(N_spline)
        self.C1_tube_radius.SetName("TubeRadius")
        tMin = 0.01444
        tMax = 0.00868
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.C1_tube_radius.SetTuple1(i, t)
        self.tubePolyData_C1 = vtk.vtkPolyData()
        self.tubePolyData_C1 = self.C1_function_src.GetOutput()
        self.tubePolyData_C1.GetPointData().AddArray(self.C1_tube_radius)
        self.tubePolyData_C1.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_C1 = vtk.vtkTubeFilter();
        self.tuber_C1.SetInputData(self.tubePolyData_C1);
        self.tuber_C1.SetNumberOfSides(6);
        self.tuber_C1.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_C1)
        lineMapper.SetScalarRange(self.tubePolyData_C1.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_C1.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_C1 = vtk.vtkActor()
        self.tubeActor_C1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C1.SetMapper(tubeMapper)

    def C2_vein(self):
        # C2 vein:
        self.C2_vein_pts = vtk.vtkPoints()
        self.C2_vein_pts.SetDataTypeToFloat()
        self.C2_vein_pts.InsertPoint(0,(-0.2386,2.5518,0.0))
        self.C2_vein_pts.InsertPoint(1,(-0.3181,2.4795,0.0))
        self.C2_vein_pts.InsertPoint(2,(-0.3904,2.4073,0.0))
        self.C2_vein_pts.InsertPoint(3,(-0.4410,2.3350,0.0))
        self.C2_vein_pts.InsertPoint(4,(-0.4916,2.2627,0.0))
        self.C2_vein_pts.InsertPoint(5,(-0.5277,2.1904,0.0))
        self.C2_vein_pts.InsertPoint(6,(-0.5566,2.1181,0.0))
        self.C2_vein_pts.InsertPoint(7,(-0.5928,2.0458,0.0))
        self.C2_vein_pts.InsertPoint(8,(-0.6145,1.9735,0.0))
        self.C2_vein_pts.InsertPoint(9,(-0.6362,1.9012,0.0))
        self.C2_vein_pts.InsertPoint(10,(-0.6578,1.8289,0.0))
        self.C2_vein_pts.InsertPoint(11,(-0.6795,1.7566,0.0))
        self.C2_vein_pts.InsertPoint(12,(-0.7012,1.6844,0.0))
        self.C2_vein_pts.InsertPoint(13,(-0.7012,1.6121,0.0))
        self.C2_vein_pts.InsertPoint(14,(-0.7012,1.5976,0.0))
        # Fit a spline to the points
        self.C2_spline = vtk.vtkParametricSpline()
        self.C2_spline.SetPoints(self.C2_vein_pts)
        self.C2_function_src = vtk.vtkParametricFunctionSource()
        self.C2_function_src.SetParametricFunction(self.C2_spline)
        self.C2_function_src.SetUResolution(self.C2_vein_pts.GetNumberOfPoints())
        self.C2_function_src.Update()
        # Tube radius
        self.C2_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.C2_function_src.GetOutput().GetNumberOfPoints()
        self.C2_tube_radius.SetNumberOfTuples(N_spline)
        self.C2_tube_radius.SetName("TubeRadius")
        tMin = 0.00868
        tMax = 0.00868
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.C2_tube_radius.SetTuple1(i, t)
        self.tubePolyData_C2 = vtk.vtkPolyData()
        self.tubePolyData_C2 = self.C2_function_src.GetOutput()
        self.tubePolyData_C2.GetPointData().AddArray(self.C2_tube_radius)
        self.tubePolyData_C2.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_C2 = vtk.vtkTubeFilter();
        self.tuber_C2.SetInputData(self.tubePolyData_C2);
        self.tuber_C2.SetNumberOfSides(6);
        self.tuber_C2.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_C2)
        lineMapper.SetScalarRange(self.tubePolyData_C2.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_C2.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_C2 = vtk.vtkActor()
        self.tubeActor_C2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C2.SetMapper(tubeMapper)

    def C3_vein(self):
        # C3 vein:
        self.C3_vein_pts = vtk.vtkPoints()
        self.C3_vein_pts.SetDataTypeToFloat()
        self.C3_vein_pts.InsertPoint(0,(-0.7012,1.5976,0.0))
        self.C3_vein_pts.InsertPoint(1,(-0.7229,1.5398,0.0))
        self.C3_vein_pts.InsertPoint(2,(-0.7446,1.4675,0.0))
        self.C3_vein_pts.InsertPoint(3,(-0.7518,1.3952,0.0))
        self.C3_vein_pts.InsertPoint(4,(-0.7663,1.3229,0.0))
        self.C3_vein_pts.InsertPoint(5,(-0.7735,1.2506,0.0))
        self.C3_vein_pts.InsertPoint(6,(-0.7807,1.1783,0.0))
        self.C3_vein_pts.InsertPoint(7,(-0.7807,1.1060,0.0))
        self.C3_vein_pts.InsertPoint(8,(-0.7807,1.0337,0.0))
        self.C3_vein_pts.InsertPoint(9,(-0.7880,0.9615,0.0))
        self.C3_vein_pts.InsertPoint(10,(-0.7880,0.8892,0.0))
        self.C3_vein_pts.InsertPoint(11,(-0.7880,0.8169,0.0))
        self.C3_vein_pts.InsertPoint(12,(-0.7807,0.7446,0.0))
        self.C3_vein_pts.InsertPoint(13,(-0.7663,0.6723,0.0))
        self.C3_vein_pts.InsertPoint(14,(-0.7590,0.6000,0.0))
        self.C3_vein_pts.InsertPoint(15,(-0.7446,0.5277,0.0))
        self.C3_vein_pts.InsertPoint(16,(-0.7229,0.4554,0.0))
        self.C3_vein_pts.InsertPoint(17,(-0.6940,0.3831,0.0))
        self.C3_vein_pts.InsertPoint(18,(-0.6434,0.3108,0.0))
        self.C3_vein_pts.InsertPoint(19,(-0.5566,0.2386,0.0))
        self.C3_vein_pts.InsertPoint(20,(-0.3831,0.1663,0.0))
        self.C3_vein_pts.InsertPoint(21,(-0.2169,0.0940,0.0))
        self.C3_vein_pts.InsertPoint(22,(-0.1157,0.0217,0.0))
        self.C3_vein_pts.InsertPoint(23,(-0.0867,0.0145,0.0))
        # Fit a spline to the point
        self.C3_spline = vtk.vtkParametricSpline()
        self.C3_spline.SetPoints(self.C3_vein_pts)
        self.C3_function_src = vtk.vtkParametricFunctionSource()
        self.C3_function_src.SetParametricFunction(self.C3_spline)
        self.C3_function_src.SetUResolution(self.C3_vein_pts.GetNumberOfPoints())
        self.C3_function_src.Update()
        # Tube radius
        self.C3_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.C3_function_src.GetOutput().GetNumberOfPoints()
        self.C3_tube_radius.SetNumberOfTuples(N_spline)
        self.C3_tube_radius.SetName("TubeRadius")
        tMin = 0.00868
        tMax = 0.00868
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.C3_tube_radius.SetTuple1(i, t)
        self.tubePolyData_C3 = vtk.vtkPolyData()
        self.tubePolyData_C3 = self.C3_function_src.GetOutput()
        self.tubePolyData_C3.GetPointData().AddArray(self.C3_tube_radius)
        self.tubePolyData_C3.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_C3 = vtk.vtkTubeFilter();
        self.tuber_C3.SetInputData(self.tubePolyData_C3);
        self.tuber_C3.SetNumberOfSides(6);
        self.tuber_C3.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_C3)
        lineMapper.SetScalarRange(self.tubePolyData_C3.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_C3.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_C3 = vtk.vtkActor()
        self.tubeActor_C3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C3.SetMapper(tubeMapper)

    def A_vein(self):
        # A vein:
        self.A_vein_pts = vtk.vtkPoints()
        self.A_vein_pts.SetDataTypeToFloat()
        self.A_vein_pts.InsertPoint(0,(0.0072,0.7157,0.0))
        self.A_vein_pts.InsertPoint(1,(-0.0795,0.6940,0.0))
        # Fit a spline to the points
        self.A_spline = vtk.vtkParametricSpline()
        self.A_spline.SetPoints(self.A_vein_pts)
        self.A_function_src = vtk.vtkParametricFunctionSource()
        self.A_function_src.SetParametricFunction(self.A_spline)
        self.A_function_src.SetUResolution(self.A_vein_pts.GetNumberOfPoints())
        self.A_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.A_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.A_function_src.GetOutput().GetNumberOfPoints()
        self.A_tube_radius.SetNumberOfTuples(N_spline)
        self.A_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.01156
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.A_tube_radius.SetTuple1(i, t)
        self.tubePolyData_A = vtk.vtkPolyData()
        self.tubePolyData_A = self.A_function_src.GetOutput()
        self.tubePolyData_A.GetPointData().AddArray(self.A_tube_radius)
        self.tubePolyData_A.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_A = vtk.vtkTubeFilter();
        self.tuber_A.SetInputData(self.tubePolyData_A);
        self.tuber_A.SetNumberOfSides(6);
        self.tuber_A.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_A)
        lineMapper.SetScalarRange(self.tubePolyData_A.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_A.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_A.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_A = vtk.vtkActor()
        self.tubeActor_A.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_A.SetMapper(tubeMapper)

    def P_vein(self):
        # P vein:
        self.P_vein_pts = vtk.vtkPoints()
        self.P_vein_pts.SetDataTypeToFloat()
        self.P_vein_pts.InsertPoint(0,(-0.1952,1.2868,0.0))
        self.P_vein_pts.InsertPoint(1,(-0.3325,1.2868,0.0))
        self.P_vein_pts.InsertPoint(2,(-0.4048,1.2578,0.0))
        # Fit a spline to the points
        self.P_spline = vtk.vtkParametricSpline()
        self.P_spline.SetPoints(self.P_vein_pts)
        self.P_function_src = vtk.vtkParametricFunctionSource()
        self.P_function_src.SetParametricFunction(self.P_spline)
        self.P_function_src.SetUResolution(self.P_vein_pts.GetNumberOfPoints())
        self.P_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.P_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.P_function_src.GetOutput().GetNumberOfPoints()
        self.P_tube_radius.SetNumberOfTuples(N_spline)
        self.P_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.01156
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.P_tube_radius.SetTuple1(i, t)
        self.tubePolyData_P = vtk.vtkPolyData()
        self.tubePolyData_P = self.P_function_src.GetOutput()
        self.tubePolyData_P.GetPointData().AddArray(self.P_tube_radius)
        self.tubePolyData_P.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_P = vtk.vtkTubeFilter();
        self.tuber_P.SetInputData(self.tubePolyData_P);
        self.tuber_P.SetNumberOfSides(6);
        self.tuber_P.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_P)
        lineMapper.SetScalarRange(self.tubePolyData_P.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_P.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_P.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_P = vtk.vtkActor()
        self.tubeActor_P.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_P.SetMapper(tubeMapper)

    def membrane_0(self):
        # Vertices
        self.mem_pts_0 = vtk.vtkPoints()
        self.mem_pts_0.SetDataTypeToFloat()
        self.mem_pts_0.InsertPoint(0,(0.0,0.0,0.0))
        self.mem_pts_0.InsertPoint(1,(0.0578,0.0145,0.0))
        self.mem_pts_0.InsertPoint(2,(0.1301,0.0940,0.0))
        self.mem_pts_0.InsertPoint(3,(0.1735,0.1663,0.0))
        self.mem_pts_0.InsertPoint(4,(0.2024,0.2386,0.0))
        self.mem_pts_0.InsertPoint(5,(0.2241,0.3108,0.0))
        self.mem_pts_0.InsertPoint(6,(0.2313,0.3831,0.0))
        self.mem_pts_0.InsertPoint(7,(0.2458,0.4554,0.0))
        self.mem_pts_0.InsertPoint(8,(0.2458,0.5277,0.0))
        self.mem_pts_0.InsertPoint(9,(0.2313,0.5711,0.0))
        self.mem_pts_0.InsertPoint(10,(0.2747,0.6723,0.0))
        self.mem_pts_0.InsertPoint(11,(0.3036,0.7446,0.0))
        self.mem_pts_0.InsertPoint(12,(0.3181,0.8169,0.0))
        self.mem_pts_0.InsertPoint(13,(0.3398,0.8892,0.0))
        self.mem_pts_0.InsertPoint(14,(0.3542,0.9615,0.0))
        self.mem_pts_0.InsertPoint(15,(0.3687,1.0337,0.0))
        self.mem_pts_0.InsertPoint(16,(0.3831,1.1060,0.0))
        self.mem_pts_0.InsertPoint(17,(0.3904,1.1783,0.0))
        self.mem_pts_0.InsertPoint(18,(0.3976,1.2506,0.0))
        self.mem_pts_0.InsertPoint(19,(0.4048,1.3229,0.0))
        self.mem_pts_0.InsertPoint(20,(0.4121,1.3952,0.0))
        self.mem_pts_0.InsertPoint(21,(0.4193,1.4675,0.0))
        self.mem_pts_0.InsertPoint(22,(0.4193,1.5398,0.0))
        self.mem_pts_0.InsertPoint(23,(0.4265,1.6121,0.0))
        self.mem_pts_0.InsertPoint(24,(0.4265,1.6844,0.0))
        self.mem_pts_0.InsertPoint(25,(0.4337,1.7566,0.0))
        self.mem_pts_0.InsertPoint(26,(0.4265,1.8289,0.0))
        self.mem_pts_0.InsertPoint(27,(0.4193,1.9012,0.0))
        self.mem_pts_0.InsertPoint(28,(0.4121,1.9735,0.0))
        self.mem_pts_0.InsertPoint(29,(0.4048,2.0458,0.0))
        self.mem_pts_0.InsertPoint(30,(0.3904,2.1181,0.0))
        self.mem_pts_0.InsertPoint(31,(0.3759,2.1904,0.0))
        self.mem_pts_0.InsertPoint(32,(0.3542,2.2627,0.0))
        self.mem_pts_0.InsertPoint(33,(0.3253,2.3350,0.0))
        self.mem_pts_0.InsertPoint(34,(0.2819,2.4073,0.0))
        self.mem_pts_0.InsertPoint(35,(0.2241,2.4795,0.0))
        self.mem_pts_0.InsertPoint(36,(0.1374,2.5518,0.0))
        self.mem_pts_0.InsertPoint(37,(0.0867,2.5880,0.0))
        self.mem_pts_0.InsertPoint(38,(0.0506,2.6097,0.0))
        self.mem_pts_0.InsertPoint(39,(0.0,2.6241,0.0))
        self.mem_pts_0.InsertPoint(40,(0.0,2.6097,0.0))
        self.mem_pts_0.InsertPoint(41,(0.0,2.5880,0.0))
        self.mem_pts_0.InsertPoint(42,(0.0,2.5518,0.0))
        self.mem_pts_0.InsertPoint(43,(0.0,2.4795,0.0))
        self.mem_pts_0.InsertPoint(44,(0.0,2.4073,0.0))
        self.mem_pts_0.InsertPoint(45,(0.0,2.3350,0.0))
        self.mem_pts_0.InsertPoint(46,(0.0,2.2627,0.0))
        self.mem_pts_0.InsertPoint(47,(0.0,2.1904,0.0))
        self.mem_pts_0.InsertPoint(48,(0.0,2.1181,0.0))
        self.mem_pts_0.InsertPoint(49,(0.0,2.0458,0.0))
        self.mem_pts_0.InsertPoint(50,(0.0,1.9735,0.0))
        self.mem_pts_0.InsertPoint(51,(0.0,1.9012,0.0))
        self.mem_pts_0.InsertPoint(52,(0.0,1.8289,0.0))
        self.mem_pts_0.InsertPoint(53,(0.0,1.7566,0.0))
        self.mem_pts_0.InsertPoint(54,(0.0,1.6844,0.0))
        self.mem_pts_0.InsertPoint(55,(0.0,1.6121,0.0))
        self.mem_pts_0.InsertPoint(56,(0.0,1.5398,0.0))
        self.mem_pts_0.InsertPoint(57,(0.0,1.4675,0.0))
        self.mem_pts_0.InsertPoint(58,(0.0,1.3952,0.0))
        self.mem_pts_0.InsertPoint(59,(0.0,1.3229,0.0))
        self.mem_pts_0.InsertPoint(60,(0.0,1.2506,0.0))
        self.mem_pts_0.InsertPoint(61,(0.0,1.1783,0.0))
        self.mem_pts_0.InsertPoint(62,(0.0,1.1060,0.0))
        self.mem_pts_0.InsertPoint(63,(0.0,1.0337,0.0))
        self.mem_pts_0.InsertPoint(64,(0.0,0.9615,0.0))
        self.mem_pts_0.InsertPoint(65,(0.0,0.8892,0.0))
        self.mem_pts_0.InsertPoint(66,(0.0,0.8169,0.0))
        self.mem_pts_0.InsertPoint(67,(0.0,0.7446,0.0))
        self.mem_pts_0.InsertPoint(68,(0.0,0.6723,0.0))
        self.mem_pts_0.InsertPoint(69,(0.0,0.5711,0.0))
        self.mem_pts_0.InsertPoint(70,(0.0,0.5277,0.0))
        self.mem_pts_0.InsertPoint(71,(0.0,0.4554,0.0))
        self.mem_pts_0.InsertPoint(72,(0.0,0.3831,0.0))
        self.mem_pts_0.InsertPoint(73,(0.0,0.3108,0.0))
        self.mem_pts_0.InsertPoint(74,(0.0,0.2386,0.0))
        self.mem_pts_0.InsertPoint(75,(0.0,0.1663,0.0))
        self.mem_pts_0.InsertPoint(76,(0.0,0.0940,0.0))
        self.mem_pts_0.InsertPoint(77,(0.0,0.0145,0.0))
        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,77)
        triangle_2.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,76)
        triangle_4.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,75)
        triangle_6.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,74)
        triangle_8.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,73)
        triangle_10.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,72)
        triangle_12.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,71)
        triangle_14.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,70)
        triangle_16.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,69)
        triangle_18.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,68)
        triangle_20.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,67)
        triangle_22.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,66)
        triangle_24.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,65)
        triangle_26.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,64)
        triangle_28.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,63)
        triangle_30.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,62)
        triangle_32.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,61)
        triangle_34.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,60)
        triangle_36.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,59)
        triangle_38.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,58)
        triangle_40.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,57)
        triangle_42.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_42)
        # triangle 43
        triangle_43 = vtk.vtkTriangle()
        triangle_43.GetPointIds().SetId(0,22)
        triangle_43.GetPointIds().SetId(1,23)
        triangle_43.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_43)
        # triangle 44
        triangle_44 = vtk.vtkTriangle()
        triangle_44.GetPointIds().SetId(0,23)
        triangle_44.GetPointIds().SetId(1,56)
        triangle_44.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_44)
        # triangle 45
        triangle_45 = vtk.vtkTriangle()
        triangle_45.GetPointIds().SetId(0,23)
        triangle_45.GetPointIds().SetId(1,24)
        triangle_45.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_45)
        # triangle 46
        triangle_46 = vtk.vtkTriangle()
        triangle_46.GetPointIds().SetId(0,24)
        triangle_46.GetPointIds().SetId(1,55)
        triangle_46.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_46)
        # triangle 47
        triangle_47 = vtk.vtkTriangle()
        triangle_47.GetPointIds().SetId(0,24)
        triangle_47.GetPointIds().SetId(1,25)
        triangle_47.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_47)
        # triangle 48
        triangle_48 = vtk.vtkTriangle()
        triangle_48.GetPointIds().SetId(0,25)
        triangle_48.GetPointIds().SetId(1,54)
        triangle_48.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_48)
        # triangle 49
        triangle_49 = vtk.vtkTriangle()
        triangle_49.GetPointIds().SetId(0,25)
        triangle_49.GetPointIds().SetId(1,26)
        triangle_49.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_49)
        # triangle 50
        triangle_50 = vtk.vtkTriangle()
        triangle_50.GetPointIds().SetId(0,26)
        triangle_50.GetPointIds().SetId(1,53)
        triangle_50.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_50)
        # triangle 51
        triangle_51 = vtk.vtkTriangle()
        triangle_51.GetPointIds().SetId(0,26)
        triangle_51.GetPointIds().SetId(1,27)
        triangle_51.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_51)
        # triangle 52
        triangle_52 = vtk.vtkTriangle()
        triangle_52.GetPointIds().SetId(0,27)
        triangle_52.GetPointIds().SetId(1,52)
        triangle_52.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_52)
        # triangle 53
        triangle_53 = vtk.vtkTriangle()
        triangle_53.GetPointIds().SetId(0,27)
        triangle_53.GetPointIds().SetId(1,28)
        triangle_53.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_53)
        # triangle 54
        triangle_54 = vtk.vtkTriangle()
        triangle_54.GetPointIds().SetId(0,28)
        triangle_54.GetPointIds().SetId(1,51)
        triangle_54.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_54)
        # triangle 55
        triangle_55 = vtk.vtkTriangle()
        triangle_55.GetPointIds().SetId(0,28)
        triangle_55.GetPointIds().SetId(1,29)
        triangle_55.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_55)
        # triangle 56
        triangle_56 = vtk.vtkTriangle()
        triangle_56.GetPointIds().SetId(0,29)
        triangle_56.GetPointIds().SetId(1,50)
        triangle_56.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_56)
        # triangle 57
        triangle_57 = vtk.vtkTriangle()
        triangle_57.GetPointIds().SetId(0,29)
        triangle_57.GetPointIds().SetId(1,30)
        triangle_57.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_57)
        # triangle 58
        triangle_58 = vtk.vtkTriangle()
        triangle_58.GetPointIds().SetId(0,30)
        triangle_58.GetPointIds().SetId(1,49)
        triangle_58.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_58)
        # triangle 59
        triangle_59 = vtk.vtkTriangle()
        triangle_59.GetPointIds().SetId(0,30)
        triangle_59.GetPointIds().SetId(1,31)
        triangle_59.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_59)
        # triangle 60
        triangle_60 = vtk.vtkTriangle()
        triangle_60.GetPointIds().SetId(0,31)
        triangle_60.GetPointIds().SetId(1,48)
        triangle_60.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_60)
        # triangle 61
        triangle_61 = vtk.vtkTriangle()
        triangle_61.GetPointIds().SetId(0,31)
        triangle_61.GetPointIds().SetId(1,32)
        triangle_61.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_61)
        # triangle 62
        triangle_62 = vtk.vtkTriangle()
        triangle_62.GetPointIds().SetId(0,32)
        triangle_62.GetPointIds().SetId(1,47)
        triangle_62.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_62)
        # triangle 63
        triangle_63 = vtk.vtkTriangle()
        triangle_63.GetPointIds().SetId(0,32)
        triangle_63.GetPointIds().SetId(1,33)
        triangle_63.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_63)
        # triangle 64
        triangle_64 = vtk.vtkTriangle()
        triangle_64.GetPointIds().SetId(0,33)
        triangle_64.GetPointIds().SetId(1,46)
        triangle_64.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_64)
        # triangle 65
        triangle_65 = vtk.vtkTriangle()
        triangle_65.GetPointIds().SetId(0,33)
        triangle_65.GetPointIds().SetId(1,34)
        triangle_65.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_65)
        # triangle 66
        triangle_66 = vtk.vtkTriangle()
        triangle_66.GetPointIds().SetId(0,34)
        triangle_66.GetPointIds().SetId(1,45)
        triangle_66.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_66)
        # triangle 67
        triangle_67 = vtk.vtkTriangle()
        triangle_67.GetPointIds().SetId(0,34)
        triangle_67.GetPointIds().SetId(1,35)
        triangle_67.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_67)
        # triangle 68
        triangle_68 = vtk.vtkTriangle()
        triangle_68.GetPointIds().SetId(0,35)
        triangle_68.GetPointIds().SetId(1,44)
        triangle_68.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_68)
        # triangle 69
        triangle_69 = vtk.vtkTriangle()
        triangle_69.GetPointIds().SetId(0,35)
        triangle_69.GetPointIds().SetId(1,36)
        triangle_69.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_69)
        # triangle 70
        triangle_70 = vtk.vtkTriangle()
        triangle_70.GetPointIds().SetId(0,36)
        triangle_70.GetPointIds().SetId(1,43)
        triangle_70.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_70)
        # triangle 71
        triangle_71 = vtk.vtkTriangle()
        triangle_71.GetPointIds().SetId(0,36)
        triangle_71.GetPointIds().SetId(1,37)
        triangle_71.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_71)
        # triangle 72
        triangle_72 = vtk.vtkTriangle()
        triangle_72.GetPointIds().SetId(0,37)
        triangle_72.GetPointIds().SetId(1,42)
        triangle_72.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_72)
        # triangle 73
        triangle_73 = vtk.vtkTriangle()
        triangle_73.GetPointIds().SetId(0,37)
        triangle_73.GetPointIds().SetId(1,38)
        triangle_73.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_73)
        # triangle 74
        triangle_74 = vtk.vtkTriangle()
        triangle_74.GetPointIds().SetId(0,38)
        triangle_74.GetPointIds().SetId(1,41)
        triangle_74.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_74)
        # triangle 75
        triangle_75 = vtk.vtkTriangle()
        triangle_75.GetPointIds().SetId(0,38)
        triangle_75.GetPointIds().SetId(1,39)
        triangle_75.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_75)

        self.mem_0 = vtk.vtkPolyData()
        self.mem_0.SetPoints(self.mem_pts_0)
        self.mem_0.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_0)

        self.membrane_Actor_0 = vtk.vtkActor()
        self.membrane_Actor_0.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_0.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_0.ForceTranslucentOn()
        self.membrane_Actor_0.SetMapper(Mapper)

    def membrane_1(self):
        # Vertices
        self.mem_pts_1 = vtk.vtkPoints()
        self.mem_pts_1.SetDataTypeToFloat()
        self.mem_pts_1.InsertPoint(0,(0.0,0.0,0.0))
        self.mem_pts_1.InsertPoint(1,(0.0,0.0145,0.0))
        self.mem_pts_1.InsertPoint(2,(0.0,0.0940,0.0))
        self.mem_pts_1.InsertPoint(3,(0.0,0.1663,0.0))
        self.mem_pts_1.InsertPoint(4,(0.0,0.2386,0.0))
        self.mem_pts_1.InsertPoint(5,(0.0,0.3108,0.0))
        self.mem_pts_1.InsertPoint(6,(0.0,0.3831,0.0))
        self.mem_pts_1.InsertPoint(7,(0.0,0.4554,0.0))
        self.mem_pts_1.InsertPoint(8,(0.0,0.5277,0.0))
        self.mem_pts_1.InsertPoint(9,(0.0,0.5711,0.0))
        self.mem_pts_1.InsertPoint(10,(0.0,0.6723,0.0))
        self.mem_pts_1.InsertPoint(11,(0.0,0.7446,0.0))
        self.mem_pts_1.InsertPoint(12,(0.0,0.8169,0.0))
        self.mem_pts_1.InsertPoint(13,(0.0,0.8892,0.0))
        self.mem_pts_1.InsertPoint(14,(0.0,0.9615,0.0))
        self.mem_pts_1.InsertPoint(15,(0.0,1.0337,0.0))
        self.mem_pts_1.InsertPoint(16,(0.0,1.1060,0.0))
        self.mem_pts_1.InsertPoint(17,(0.0,1.1783,0.0))
        self.mem_pts_1.InsertPoint(18,(0.0,1.2506,0.0))
        self.mem_pts_1.InsertPoint(19,(0.0,1.3229,0.0))
        self.mem_pts_1.InsertPoint(20,(0.0,1.3952,0.0))
        self.mem_pts_1.InsertPoint(21,(0.0,1.4675,0.0))
        self.mem_pts_1.InsertPoint(22,(0.0,1.5398,0.0))
        self.mem_pts_1.InsertPoint(23,(0.0,1.6121,0.0))
        self.mem_pts_1.InsertPoint(24,(0.0,1.6844,0.0))
        self.mem_pts_1.InsertPoint(25,(0.0,1.7566,0.0))
        self.mem_pts_1.InsertPoint(26,(0.0,1.8289,0.0))
        self.mem_pts_1.InsertPoint(27,(0.0,1.9012,0.0))
        self.mem_pts_1.InsertPoint(28,(0.0,1.9735,0.0))
        self.mem_pts_1.InsertPoint(29,(0.0,2.0458,0.0))
        self.mem_pts_1.InsertPoint(30,(0.0,2.1181,0.0))
        self.mem_pts_1.InsertPoint(31,(0.0,2.1904,0.0))
        self.mem_pts_1.InsertPoint(32,(0.0,2.2627,0.0))
        self.mem_pts_1.InsertPoint(33,(0.0,2.3350,0.0))
        self.mem_pts_1.InsertPoint(34,(0.0,2.4073,0.0))
        self.mem_pts_1.InsertPoint(35,(0.0,2.4795,0.0))
        self.mem_pts_1.InsertPoint(36,(0.0,2.5518,0.0))
        self.mem_pts_1.InsertPoint(37,(0.0,2.5880,0.0))
        self.mem_pts_1.InsertPoint(38,(0.0,2.6097,0.0))
        self.mem_pts_1.InsertPoint(39,(0.0,2.6241,0.0)) # tip
        self.mem_pts_1.InsertPoint(40,(-0.1301,2.6097,0.0))
        self.mem_pts_1.InsertPoint(41,(-0.1880,2.5880,0.0))
        self.mem_pts_1.InsertPoint(42,(-0.2313,2.5518,0.0))
        self.mem_pts_1.InsertPoint(43,(-0.2169,2.4795,0.0))
        self.mem_pts_1.InsertPoint(44,(-0.2169,2.4073,0.0))
        self.mem_pts_1.InsertPoint(45,(-0.2096,2.3350,0.0))
        self.mem_pts_1.InsertPoint(46,(-0.2096,2.2627,0.0))
        self.mem_pts_1.InsertPoint(47,(-0.2024,2.1904,0.0))
        self.mem_pts_1.InsertPoint(48,(-0.2024,2.1181,0.0))
        self.mem_pts_1.InsertPoint(49,(-0.1952,2.0458,0.0))
        self.mem_pts_1.InsertPoint(50,(-0.1952,1.9735,0.0))
        self.mem_pts_1.InsertPoint(51,(-0.1876,1.9012,0.0))
        self.mem_pts_1.InsertPoint(52,(-0.1876,1.8289,0.0))
        self.mem_pts_1.InsertPoint(53,(-0.1807,1.7566,0.0))
        self.mem_pts_1.InsertPoint(54,(-0.1807,1.6844,0.0))
        self.mem_pts_1.InsertPoint(55,(-0.1735,1.6121,0.0))
        self.mem_pts_1.InsertPoint(56,(-0.1735,1.5398,0.0))
        self.mem_pts_1.InsertPoint(57,(-0.1663,1.4675,0.0))
        self.mem_pts_1.InsertPoint(58,(-0.1663,1.3952,0.0))
        self.mem_pts_1.InsertPoint(59,(-0.1590,1.3229,0.0))
        self.mem_pts_1.InsertPoint(60,(-0.1590,1.2506,0.0))
        self.mem_pts_1.InsertPoint(61,(-0.1518,1.1783,0.0))
        self.mem_pts_1.InsertPoint(62,(-0.1518,1.1060,0.0))
        self.mem_pts_1.InsertPoint(63,(-0.1446,1.0337,0.0))
        self.mem_pts_1.InsertPoint(64,(-0.1446,0.9615,0.0))
        self.mem_pts_1.InsertPoint(65,(-0.1374,0.8892,0.0))
        self.mem_pts_1.InsertPoint(66,(-0.1374,0.8169,0.0))
        self.mem_pts_1.InsertPoint(67,(-0.1301,0.7446,0.0))
        self.mem_pts_1.InsertPoint(68,(-0.1301,0.6723,0.0))
        self.mem_pts_1.InsertPoint(69,(-0.1229,0.5711,0.0))
        self.mem_pts_1.InsertPoint(70,(-0.1229,0.5277,0.0))
        self.mem_pts_1.InsertPoint(71,(-0.1157,0.4554,0.0))
        self.mem_pts_1.InsertPoint(72,(-0.1157,0.3831,0.0))
        self.mem_pts_1.InsertPoint(73,(-0.1084,0.3108,0.0))
        self.mem_pts_1.InsertPoint(74,(-0.1084,0.2386,0.0))
        self.mem_pts_1.InsertPoint(75,(-0.1012,0.1663,0.0))
        self.mem_pts_1.InsertPoint(76,(-0.1012,0.0940,0.0))
        self.mem_pts_1.InsertPoint(77,(-0.0940,0.0145,0.0))
        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,77)
        triangle_2.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,76)
        triangle_4.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,75)
        triangle_6.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,74)
        triangle_8.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,73)
        triangle_10.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,72)
        triangle_12.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,71)
        triangle_14.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,70)
        triangle_16.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,69)
        triangle_18.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,68)
        triangle_20.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,67)
        triangle_22.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,66)
        triangle_24.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,65)
        triangle_26.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,64)
        triangle_28.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,63)
        triangle_30.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,62)
        triangle_32.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,61)
        triangle_34.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,60)
        triangle_36.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,59)
        triangle_38.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,58)
        triangle_40.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,57)
        triangle_42.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_42)
        # triangle 43
        triangle_43 = vtk.vtkTriangle()
        triangle_43.GetPointIds().SetId(0,22)
        triangle_43.GetPointIds().SetId(1,23)
        triangle_43.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_43)
        # triangle 44
        triangle_44 = vtk.vtkTriangle()
        triangle_44.GetPointIds().SetId(0,23)
        triangle_44.GetPointIds().SetId(1,56)
        triangle_44.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_44)
        # triangle 45
        triangle_45 = vtk.vtkTriangle()
        triangle_45.GetPointIds().SetId(0,23)
        triangle_45.GetPointIds().SetId(1,24)
        triangle_45.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_45)
        # triangle 46
        triangle_46 = vtk.vtkTriangle()
        triangle_46.GetPointIds().SetId(0,24)
        triangle_46.GetPointIds().SetId(1,55)
        triangle_46.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_46)
        # triangle 47
        triangle_47 = vtk.vtkTriangle()
        triangle_47.GetPointIds().SetId(0,24)
        triangle_47.GetPointIds().SetId(1,25)
        triangle_47.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_47)
        # triangle 48
        triangle_48 = vtk.vtkTriangle()
        triangle_48.GetPointIds().SetId(0,25)
        triangle_48.GetPointIds().SetId(1,54)
        triangle_48.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_48)
        # triangle 49
        triangle_49 = vtk.vtkTriangle()
        triangle_49.GetPointIds().SetId(0,25)
        triangle_49.GetPointIds().SetId(1,26)
        triangle_49.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_49)
        # triangle 50
        triangle_50 = vtk.vtkTriangle()
        triangle_50.GetPointIds().SetId(0,26)
        triangle_50.GetPointIds().SetId(1,53)
        triangle_50.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_50)
        # triangle 51
        triangle_51 = vtk.vtkTriangle()
        triangle_51.GetPointIds().SetId(0,26)
        triangle_51.GetPointIds().SetId(1,27)
        triangle_51.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_51)
        # triangle 52
        triangle_52 = vtk.vtkTriangle()
        triangle_52.GetPointIds().SetId(0,27)
        triangle_52.GetPointIds().SetId(1,52)
        triangle_52.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_52)
        # triangle 53
        triangle_53 = vtk.vtkTriangle()
        triangle_53.GetPointIds().SetId(0,27)
        triangle_53.GetPointIds().SetId(1,28)
        triangle_53.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_53)
        # triangle 54
        triangle_54 = vtk.vtkTriangle()
        triangle_54.GetPointIds().SetId(0,28)
        triangle_54.GetPointIds().SetId(1,51)
        triangle_54.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_54)
        # triangle 55
        triangle_55 = vtk.vtkTriangle()
        triangle_55.GetPointIds().SetId(0,28)
        triangle_55.GetPointIds().SetId(1,29)
        triangle_55.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_55)
        # triangle 56
        triangle_56 = vtk.vtkTriangle()
        triangle_56.GetPointIds().SetId(0,29)
        triangle_56.GetPointIds().SetId(1,50)
        triangle_56.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_56)
        # triangle 57
        triangle_57 = vtk.vtkTriangle()
        triangle_57.GetPointIds().SetId(0,29)
        triangle_57.GetPointIds().SetId(1,30)
        triangle_57.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_57)
        # triangle 58
        triangle_58 = vtk.vtkTriangle()
        triangle_58.GetPointIds().SetId(0,30)
        triangle_58.GetPointIds().SetId(1,49)
        triangle_58.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_58)
        # triangle 59
        triangle_59 = vtk.vtkTriangle()
        triangle_59.GetPointIds().SetId(0,30)
        triangle_59.GetPointIds().SetId(1,31)
        triangle_59.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_59)
        # triangle 60
        triangle_60 = vtk.vtkTriangle()
        triangle_60.GetPointIds().SetId(0,31)
        triangle_60.GetPointIds().SetId(1,48)
        triangle_60.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_60)
        # triangle 61
        triangle_61 = vtk.vtkTriangle()
        triangle_61.GetPointIds().SetId(0,31)
        triangle_61.GetPointIds().SetId(1,32)
        triangle_61.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_61)
        # triangle 62
        triangle_62 = vtk.vtkTriangle()
        triangle_62.GetPointIds().SetId(0,32)
        triangle_62.GetPointIds().SetId(1,47)
        triangle_62.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_62)
        # triangle 63
        triangle_63 = vtk.vtkTriangle()
        triangle_63.GetPointIds().SetId(0,32)
        triangle_63.GetPointIds().SetId(1,33)
        triangle_63.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_63)
        # triangle 64
        triangle_64 = vtk.vtkTriangle()
        triangle_64.GetPointIds().SetId(0,33)
        triangle_64.GetPointIds().SetId(1,46)
        triangle_64.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_64)
        # triangle 65
        triangle_65 = vtk.vtkTriangle()
        triangle_65.GetPointIds().SetId(0,33)
        triangle_65.GetPointIds().SetId(1,34)
        triangle_65.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_65)
        # triangle 66
        triangle_66 = vtk.vtkTriangle()
        triangle_66.GetPointIds().SetId(0,34)
        triangle_66.GetPointIds().SetId(1,45)
        triangle_66.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_66)
        # triangle 67
        triangle_67 = vtk.vtkTriangle()
        triangle_67.GetPointIds().SetId(0,34)
        triangle_67.GetPointIds().SetId(1,35)
        triangle_67.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_67)
        # triangle 68
        triangle_68 = vtk.vtkTriangle()
        triangle_68.GetPointIds().SetId(0,35)
        triangle_68.GetPointIds().SetId(1,44)
        triangle_68.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_68)
        # triangle 69
        triangle_69 = vtk.vtkTriangle()
        triangle_69.GetPointIds().SetId(0,35)
        triangle_69.GetPointIds().SetId(1,36)
        triangle_69.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_69)
        # triangle 70
        triangle_70 = vtk.vtkTriangle()
        triangle_70.GetPointIds().SetId(0,36)
        triangle_70.GetPointIds().SetId(1,43)
        triangle_70.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_70)
        # triangle 71
        triangle_71 = vtk.vtkTriangle()
        triangle_71.GetPointIds().SetId(0,36)
        triangle_71.GetPointIds().SetId(1,37)
        triangle_71.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_71)
        # triangle 72
        triangle_72 = vtk.vtkTriangle()
        triangle_72.GetPointIds().SetId(0,37)
        triangle_72.GetPointIds().SetId(1,42)
        triangle_72.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_72)
        # triangle 73
        triangle_73 = vtk.vtkTriangle()
        triangle_73.GetPointIds().SetId(0,37)
        triangle_73.GetPointIds().SetId(1,38)
        triangle_73.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_73)
        # triangle 74
        triangle_74 = vtk.vtkTriangle()
        triangle_74.GetPointIds().SetId(0,38)
        triangle_74.GetPointIds().SetId(1,41)
        triangle_74.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_74)
        # triangle 75
        triangle_75 = vtk.vtkTriangle()
        triangle_75.GetPointIds().SetId(0,38)
        triangle_75.GetPointIds().SetId(1,39)
        triangle_75.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_75)

        self.mem_1 = vtk.vtkPolyData()
        self.mem_1.SetPoints(self.mem_pts_1)
        self.mem_1.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_1)

        self.membrane_Actor_1 = vtk.vtkActor()
        self.membrane_Actor_1.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_1.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_1.ForceTranslucentOn()
        self.membrane_Actor_1.SetMapper(Mapper)

    def membrane_2(self):
        # Vertices
        self.mem_pts_2 = vtk.vtkPoints()
        self.mem_pts_2.SetDataTypeToFloat()
        self.mem_pts_2.InsertPoint(0,(-0.0867,0.0145,0.0))
        self.mem_pts_2.InsertPoint(1,(-0.1157,0.0940,0.0))
        self.mem_pts_2.InsertPoint(2,(-0.1518,0.1663,0.0))
        self.mem_pts_2.InsertPoint(3,(-0.1735,0.2386,0.0))
        self.mem_pts_2.InsertPoint(4,(-0.2024,0.3108,0.0))
        self.mem_pts_2.InsertPoint(5,(-0.2313,0.3831,0.0))
        self.mem_pts_2.InsertPoint(6,(-0.2530,0.4554,0.0))
        self.mem_pts_2.InsertPoint(7,(-0.2819,0.5277,0.0))
        self.mem_pts_2.InsertPoint(8,(-0.3109,0.5711,0.0))
        self.mem_pts_2.InsertPoint(9,(-0.3398,0.6723,0.0))
        self.mem_pts_2.InsertPoint(10,(-0.3687,0.7446,0.0))
        self.mem_pts_2.InsertPoint(11,(-0.3976,0.8169,0.0))
        self.mem_pts_2.InsertPoint(12,(-0.4193,0.8892,0.0))
        self.mem_pts_2.InsertPoint(13,(-0.4554,0.9615,0.0))
        self.mem_pts_2.InsertPoint(14,(-0.4771,1.0337,0.0))
        self.mem_pts_2.InsertPoint(15,(-0.5060,1.1060,0.0))
        self.mem_pts_2.InsertPoint(16,(-0.5350,1.1783,0.0))
        self.mem_pts_2.InsertPoint(17,(-0.5566,1.2506,0.0))
        self.mem_pts_2.InsertPoint(18,(-0.5856,1.3229,0.0))
        self.mem_pts_2.InsertPoint(19,(-0.6145,1.3952,0.0))
        self.mem_pts_2.InsertPoint(20,(-0.6434,1.4675,0.0))
        self.mem_pts_2.InsertPoint(21,(-0.6723,1.5398,0.0))
        self.mem_pts_2.InsertPoint(22,(-0.6940,1.6121,0.0))
        self.mem_pts_2.InsertPoint(23,(-0.6940,1.6844,0.0))
        self.mem_pts_2.InsertPoint(24,(-0.6868,1.7566,0.0))
        self.mem_pts_2.InsertPoint(25,(-0.6578,1.8289,0.0))
        self.mem_pts_2.InsertPoint(26,(-0.6362,1.9012,0.0))
        self.mem_pts_2.InsertPoint(27,(-0.6145,1.9735,0.0))
        self.mem_pts_2.InsertPoint(28,(-0.5856,2.0458,0.0))
        self.mem_pts_2.InsertPoint(29,(-0.5566,2.1181,0.0))
        self.mem_pts_2.InsertPoint(30,(-0.5205,2.1904,0.0))
        self.mem_pts_2.InsertPoint(31,(-0.4843,2.2627,0.0))
        self.mem_pts_2.InsertPoint(32,(-0.4337,2.3350,0.0))
        self.mem_pts_2.InsertPoint(33,(-0.3831,2.4073,0.0))
        self.mem_pts_2.InsertPoint(34,(-0.2313,2.5518,0.0))
        self.mem_pts_2.InsertPoint(35,(-0.2169,2.4795,0.0))
        self.mem_pts_2.InsertPoint(36,(-0.2169,2.4073,0.0))
        self.mem_pts_2.InsertPoint(37,(-0.2096,2.3350,0.0))
        self.mem_pts_2.InsertPoint(38,(-0.2096,2.2627,0.0))
        self.mem_pts_2.InsertPoint(39,(-0.2024,2.1904,0.0))
        self.mem_pts_2.InsertPoint(40,(-0.2024,2.1181,0.0))
        self.mem_pts_2.InsertPoint(41,(-0.1952,2.0458,0.0))
        self.mem_pts_2.InsertPoint(42,(-0.1952,1.9735,0.0))
        self.mem_pts_2.InsertPoint(43,(-0.1876,1.9012,0.0))
        self.mem_pts_2.InsertPoint(44,(-0.1876,1.8289,0.0))
        self.mem_pts_2.InsertPoint(45,(-0.1807,1.7566,0.0))
        self.mem_pts_2.InsertPoint(46,(-0.1807,1.6844,0.0))
        self.mem_pts_2.InsertPoint(47,(-0.1735,1.6121,0.0))
        self.mem_pts_2.InsertPoint(48,(-0.1735,1.5398,0.0))
        self.mem_pts_2.InsertPoint(49,(-0.1663,1.4675,0.0))
        self.mem_pts_2.InsertPoint(50,(-0.1663,1.3952,0.0))
        self.mem_pts_2.InsertPoint(51,(-0.1590,1.3229,0.0))
        self.mem_pts_2.InsertPoint(52,(-0.1590,1.2506,0.0))
        self.mem_pts_2.InsertPoint(53,(-0.1518,1.1783,0.0))
        self.mem_pts_2.InsertPoint(54,(-0.1518,1.1060,0.0))
        self.mem_pts_2.InsertPoint(55,(-0.1446,1.0337,0.0))
        self.mem_pts_2.InsertPoint(56,(-0.1446,0.9615,0.0))
        self.mem_pts_2.InsertPoint(57,(-0.1374,0.8892,0.0))
        self.mem_pts_2.InsertPoint(58,(-0.1374,0.8169,0.0))
        self.mem_pts_2.InsertPoint(59,(-0.1301,0.7446,0.0))
        self.mem_pts_2.InsertPoint(60,(-0.1301,0.6723,0.0))
        self.mem_pts_2.InsertPoint(61,(-0.1229,0.5711,0.0))
        self.mem_pts_2.InsertPoint(62,(-0.1229,0.5277,0.0))
        self.mem_pts_2.InsertPoint(63,(-0.1157,0.4554,0.0))
        self.mem_pts_2.InsertPoint(64,(-0.1157,0.3831,0.0))
        self.mem_pts_2.InsertPoint(65,(-0.1084,0.3108,0.0))
        self.mem_pts_2.InsertPoint(66,(-0.1084,0.2386,0.0))
        self.mem_pts_2.InsertPoint(67,(-0.1012,0.1663,0.0))
        self.mem_pts_2.InsertPoint(68,(-0.1012,0.0940,0.0))

        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,68)
        triangle_2.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,67)
        triangle_4.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,66)
        triangle_6.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,65)
        triangle_8.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,64)
        triangle_10.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,63)
        triangle_12.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,62)
        triangle_14.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,61)
        triangle_16.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,60)
        triangle_18.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,59)
        triangle_20.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,58)
        triangle_22.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,57)
        triangle_24.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,56)
        triangle_26.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,55)
        triangle_28.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,54)
        triangle_30.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,53)
        triangle_32.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,52)
        triangle_34.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,51)
        triangle_36.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,50)
        triangle_38.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,49)
        triangle_40.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,48)
        triangle_42.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_42)
        # triangle 43
        triangle_43 = vtk.vtkTriangle()
        triangle_43.GetPointIds().SetId(0,22)
        triangle_43.GetPointIds().SetId(1,23)
        triangle_43.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_43)
        # triangle 44
        triangle_44 = vtk.vtkTriangle()
        triangle_44.GetPointIds().SetId(0,23)
        triangle_44.GetPointIds().SetId(1,47)
        triangle_44.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_44)
        # triangle 45
        triangle_45 = vtk.vtkTriangle()
        triangle_45.GetPointIds().SetId(0,23)
        triangle_45.GetPointIds().SetId(1,24)
        triangle_45.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_45)
        # triangle 46
        triangle_46 = vtk.vtkTriangle()
        triangle_46.GetPointIds().SetId(0,24)
        triangle_46.GetPointIds().SetId(1,46)
        triangle_46.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_46)
        # triangle 47
        triangle_47 = vtk.vtkTriangle()
        triangle_47.GetPointIds().SetId(0,24)
        triangle_47.GetPointIds().SetId(1,25)
        triangle_47.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_47)
        # triangle 48
        triangle_48 = vtk.vtkTriangle()
        triangle_48.GetPointIds().SetId(0,25)
        triangle_48.GetPointIds().SetId(1,45)
        triangle_48.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_48)
        # triangle 49
        triangle_49 = vtk.vtkTriangle()
        triangle_49.GetPointIds().SetId(0,25)
        triangle_49.GetPointIds().SetId(1,26)
        triangle_49.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_49)
        # triangle 50
        triangle_50 = vtk.vtkTriangle()
        triangle_50.GetPointIds().SetId(0,26)
        triangle_50.GetPointIds().SetId(1,44)
        triangle_50.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_50)
        # triangle 51
        triangle_51 = vtk.vtkTriangle()
        triangle_51.GetPointIds().SetId(0,26)
        triangle_51.GetPointIds().SetId(1,27)
        triangle_51.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_51)
        # triangle 52
        triangle_52 = vtk.vtkTriangle()
        triangle_52.GetPointIds().SetId(0,27)
        triangle_52.GetPointIds().SetId(1,43)
        triangle_52.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_52)
        # triangle 53
        triangle_53 = vtk.vtkTriangle()
        triangle_53.GetPointIds().SetId(0,27)
        triangle_53.GetPointIds().SetId(1,28)
        triangle_53.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_53)
        # triangle 54
        triangle_54 = vtk.vtkTriangle()
        triangle_54.GetPointIds().SetId(0,28)
        triangle_54.GetPointIds().SetId(1,42)
        triangle_54.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_54)
        # triangle 55
        triangle_55 = vtk.vtkTriangle()
        triangle_55.GetPointIds().SetId(0,28)
        triangle_55.GetPointIds().SetId(1,29)
        triangle_55.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_55)
        # triangle 56
        triangle_56 = vtk.vtkTriangle()
        triangle_56.GetPointIds().SetId(0,29)
        triangle_56.GetPointIds().SetId(1,41)
        triangle_56.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_56)
        # triangle 57
        triangle_57 = vtk.vtkTriangle()
        triangle_57.GetPointIds().SetId(0,29)
        triangle_57.GetPointIds().SetId(1,30)
        triangle_57.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_57)
        # triangle 58
        triangle_58 = vtk.vtkTriangle()
        triangle_58.GetPointIds().SetId(0,30)
        triangle_58.GetPointIds().SetId(1,40)
        triangle_58.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_58)
        # triangle 59
        triangle_59 = vtk.vtkTriangle()
        triangle_59.GetPointIds().SetId(0,30)
        triangle_59.GetPointIds().SetId(1,31)
        triangle_59.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_59)
        # triangle 60
        triangle_60 = vtk.vtkTriangle()
        triangle_60.GetPointIds().SetId(0,31)
        triangle_60.GetPointIds().SetId(1,39)
        triangle_60.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_60)
        # triangle 61
        triangle_61 = vtk.vtkTriangle()
        triangle_61.GetPointIds().SetId(0,31)
        triangle_61.GetPointIds().SetId(1,32)
        triangle_61.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_61)
        # triangle 62
        triangle_62 = vtk.vtkTriangle()
        triangle_62.GetPointIds().SetId(0,32)
        triangle_62.GetPointIds().SetId(1,38)
        triangle_62.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_62)
        # triangle 63
        triangle_63 = vtk.vtkTriangle()
        triangle_63.GetPointIds().SetId(0,32)
        triangle_63.GetPointIds().SetId(1,33)
        triangle_63.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_63)
        # triangle 64
        triangle_64 = vtk.vtkTriangle()
        triangle_64.GetPointIds().SetId(0,33)
        triangle_64.GetPointIds().SetId(1,37)
        triangle_64.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_64)
        # triangle 65
        triangle_65 = vtk.vtkTriangle()
        triangle_65.GetPointIds().SetId(0,33)
        triangle_65.GetPointIds().SetId(1,34)
        triangle_65.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_65)
        # triangle 66
        triangle_66 = vtk.vtkTriangle()
        triangle_66.GetPointIds().SetId(0,34)
        triangle_66.GetPointIds().SetId(1,36)
        triangle_66.GetPointIds().SetId(2,35)
        triangles.InsertNextCell(triangle_66)
        # triangle 67
        triangle_67 = vtk.vtkTriangle()
        triangle_67.GetPointIds().SetId(0,34)
        triangle_67.GetPointIds().SetId(1,35)
        triangle_67.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_67)        

        self.mem_2 = vtk.vtkPolyData()
        self.mem_2.SetPoints(self.mem_pts_2)
        self.mem_2.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_2)

        self.membrane_Actor_2 = vtk.vtkActor()
        self.membrane_Actor_2.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_2.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_2.ForceTranslucentOn()
        self.membrane_Actor_2.SetMapper(Mapper)

    def membrane_3(self):
        # Vertices
        self.mem_pts_3 = vtk.vtkPoints()
        self.mem_pts_3.SetDataTypeToFloat()
        self.mem_pts_3.InsertPoint(0,(-0.0867,0.0145,0.0))
        self.mem_pts_3.InsertPoint(1,(-0.1157,0.0940,0.0))
        self.mem_pts_3.InsertPoint(2,(-0.1518,0.1663,0.0))
        self.mem_pts_3.InsertPoint(3,(-0.1735,0.2386,0.0))
        self.mem_pts_3.InsertPoint(4,(-0.2024,0.3108,0.0))
        self.mem_pts_3.InsertPoint(5,(-0.2313,0.3831,0.0))
        self.mem_pts_3.InsertPoint(6,(-0.2530,0.4554,0.0))
        self.mem_pts_3.InsertPoint(7,(-0.2819,0.5277,0.0))
        self.mem_pts_3.InsertPoint(8,(-0.3109,0.5711,0.0))
        self.mem_pts_3.InsertPoint(9,(-0.3398,0.6723,0.0))
        self.mem_pts_3.InsertPoint(10,(-0.3687,0.7446,0.0))
        self.mem_pts_3.InsertPoint(11,(-0.3976,0.8169,0.0))
        self.mem_pts_3.InsertPoint(12,(-0.4193,0.8892,0.0))
        self.mem_pts_3.InsertPoint(13,(-0.4554,0.9615,0.0))
        self.mem_pts_3.InsertPoint(14,(-0.4771,1.0337,0.0))
        self.mem_pts_3.InsertPoint(15,(-0.5060,1.1060,0.0))
        self.mem_pts_3.InsertPoint(16,(-0.5350,1.1783,0.0))
        self.mem_pts_3.InsertPoint(17,(-0.5566,1.2506,0.0))
        self.mem_pts_3.InsertPoint(18,(-0.5856,1.3229,0.0))
        self.mem_pts_3.InsertPoint(19,(-0.6145,1.3952,0.0))
        self.mem_pts_3.InsertPoint(20,(-0.6434,1.4675,0.0))
        self.mem_pts_3.InsertPoint(21,(-0.6723,1.5398,0.0))
        self.mem_pts_3.InsertPoint(22,(-0.6940,1.6121,0.0))
        self.mem_pts_3.InsertPoint(23,(-0.7229,1.5398,0.0))
        self.mem_pts_3.InsertPoint(24,(-0.7446,1.4675,0.0))
        self.mem_pts_3.InsertPoint(25,(-0.7518,1.3952,0.0))
        self.mem_pts_3.InsertPoint(26,(-0.7663,1.3229,0.0))
        self.mem_pts_3.InsertPoint(27,(-0.7735,1.2506,0.0))
        self.mem_pts_3.InsertPoint(28,(-0.7807,1.1783,0.0))
        self.mem_pts_3.InsertPoint(29,(-0.7807,1.1060,0.0))
        self.mem_pts_3.InsertPoint(30,(-0.7807,1.0337,0.0))
        self.mem_pts_3.InsertPoint(31,(-0.7880,0.9615,0.0))
        self.mem_pts_3.InsertPoint(32,(-0.7880,0.8892,0.0))
        self.mem_pts_3.InsertPoint(33,(-0.7880,0.8169,0.0))
        self.mem_pts_3.InsertPoint(34,(-0.7807,0.7446,0.0))
        self.mem_pts_3.InsertPoint(35,(-0.7663,0.6723,0.0))
        self.mem_pts_3.InsertPoint(36,(-0.7590,0.6000,0.0))
        self.mem_pts_3.InsertPoint(37,(-0.7446,0.5277,0.0))
        self.mem_pts_3.InsertPoint(38,(-0.7229,0.4554,0.0))
        self.mem_pts_3.InsertPoint(39,(-0.6940,0.3831,0.0))
        self.mem_pts_3.InsertPoint(40,(-0.6434,0.3108,0.0))
        self.mem_pts_3.InsertPoint(41,(-0.5566,0.2386,0.0))
        self.mem_pts_3.InsertPoint(42,(-0.3831,0.1663,0.0))
        self.mem_pts_3.InsertPoint(43,(-0.2169,0.0940,0.0))
        self.mem_pts_3.InsertPoint(44,(-0.1157,0.0217,0.0))

        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,44)
        triangle_2.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,43)
        triangle_4.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,42)
        triangle_6.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,41)
        triangle_8.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,40)
        triangle_10.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,39)
        triangle_12.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,38)
        triangle_14.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,37)
        triangle_16.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,36)
        triangle_18.GetPointIds().SetId(2,35)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,35)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,35)
        triangle_20.GetPointIds().SetId(2,34)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,34)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,34)
        triangle_22.GetPointIds().SetId(2,33)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,33)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,33)
        triangle_24.GetPointIds().SetId(2,32)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,32)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,32)
        triangle_26.GetPointIds().SetId(2,31)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,31)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,31)
        triangle_28.GetPointIds().SetId(2,30)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,30)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,30)
        triangle_30.GetPointIds().SetId(2,29)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,29)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,29)
        triangle_32.GetPointIds().SetId(2,28)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,28)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,28)
        triangle_34.GetPointIds().SetId(2,27)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,27)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,27)
        triangle_36.GetPointIds().SetId(2,26)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,26)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,26)
        triangle_38.GetPointIds().SetId(2,25)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,25)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,25)
        triangle_40.GetPointIds().SetId(2,24)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,24)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,24)
        triangle_42.GetPointIds().SetId(2,23)
        triangles.InsertNextCell(triangle_42)

        self.mem_3 = vtk.vtkPolyData()
        self.mem_3.SetPoints(self.mem_pts_3)
        self.mem_3.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_3)

        self.membrane_Actor_3 = vtk.vtkActor()
        self.membrane_Actor_3.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_3.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_3.ForceTranslucentOn()
        self.membrane_Actor_3.SetMapper(Mapper)

    def root_trace(self,root_pts_in,N_pts_display):
        N_root_pts = len(root_pts_in)
        if N_root_pts>1:
            self.root_pts = vtk.vtkPoints()
            self.root_pts.SetDataTypeToFloat()
            # add points:
            if N_root_pts > N_pts_display:
                for i in range(N_root_pts-N_pts_display,N_root_pts):
                    self.root_pts.InsertNextPoint(root_pts_in[i][0],root_pts_in[i][1],root_pts_in[i][2])
            else:
                for i in range(1,N_root_pts):
                    self.root_pts.InsertNextPoint(root_pts_in[i][0],root_pts_in[i][1],root_pts_in[i][2])
            # root spline
            self.root_spline = vtk.vtkParametricSpline()
            self.root_spline.SetPoints(self.root_pts)
            self.root_function_src = vtk.vtkParametricFunctionSource()
            self.root_function_src.SetParametricFunction(self.root_spline)
            self.root_function_src.SetUResolution(self.root_pts.GetNumberOfPoints())
            self.root_function_src.Update()
            # Radius interpolation
            self.root_radius_interp = vtk.vtkTupleInterpolator()
            self.root_radius_interp.SetInterpolationTypeToLinear()
            self.root_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.root_radius = vtk.vtkDoubleArray()
            N_spline = self.root_function_src.GetOutput().GetNumberOfPoints()
            self.root_radius.SetNumberOfTuples(N_spline)
            self.root_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.root_radius.SetTuple1(i, t)
            self.tubePolyData_root = vtk.vtkPolyData()
            self.tubePolyData_root = self.root_function_src.GetOutput()
            self.tubePolyData_root.GetPointData().AddArray(self.root_radius)
            self.tubePolyData_root.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_root = vtk.vtkTubeFilter()
            self.tuber_root.SetInputData(self.tubePolyData_root)
            self.tuber_root.SetNumberOfSides(6)
            self.tuber_root.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_root)
            lineMapper.SetScalarRange(self.tubePolyData_root.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_root.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            #self.tubeActor_root = vtk.vtkActor()
            self.tubeActor_root.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_root.SetMapper(tubeMapper)
        else:
            self.root_pts = vtk.vtkPoints()
            self.root_pts.SetDataTypeToFloat()
            # add dummy points:
            self.root_pts.InsertPoint(0,(0.0,0.0,0.0))
            self.root_pts.InsertPoint(1,(0.0,0.0,0.04))
            # root spline
            self.root_spline = vtk.vtkParametricSpline()
            self.root_spline.SetPoints(self.root_pts)
            self.root_function_src = vtk.vtkParametricFunctionSource()
            self.root_function_src.SetParametricFunction(self.root_spline)
            self.root_function_src.SetUResolution(self.root_pts.GetNumberOfPoints())
            self.root_function_src.Update()
            # Radius interpolation
            self.root_radius_interp = vtk.vtkTupleInterpolator()
            self.root_radius_interp.SetInterpolationTypeToLinear()
            self.root_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.root_radius = vtk.vtkDoubleArray()
            N_spline = self.root_function_src.GetOutput().GetNumberOfPoints()
            self.root_radius.SetNumberOfTuples(N_spline)
            self.root_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.root_radius.SetTuple1(i, t)
            self.tubePolyData_root = vtk.vtkPolyData()
            self.tubePolyData_root = self.root_function_src.GetOutput()
            self.tubePolyData_root.GetPointData().AddArray(self.root_radius)
            self.tubePolyData_root.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_root = vtk.vtkTubeFilter()
            self.tuber_root.SetInputData(self.tubePolyData_root)
            self.tuber_root.SetNumberOfSides(6)
            self.tuber_root.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_root)
            lineMapper.SetScalarRange(self.tubePolyData_root.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_root.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            self.tubeActor_root = vtk.vtkActor()
            self.tubeActor_root.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_root.SetMapper(tubeMapper)

    def tip_trace(self,tip_pts_in,N_pts_display):
        N_tip_pts = len(tip_pts_in)
        if N_tip_pts>1:
            self.tip_pts = vtk.vtkPoints()
            self.tip_pts.SetDataTypeToFloat()
            # add points:
            if N_tip_pts > N_pts_display:
                for i in range(N_tip_pts-N_pts_display,N_tip_pts):
                    self.tip_pts.InsertNextPoint(tip_pts_in[i][0],tip_pts_in[i][1],tip_pts_in[i][2])
            else:
                for i in range(1,N_tip_pts):
                    self.tip_pts.InsertNextPoint(tip_pts_in[i][0],tip_pts_in[i][1],tip_pts_in[i][2])
            # root spline
            self.tip_spline = vtk.vtkParametricSpline()
            self.tip_spline.SetPoints(self.tip_pts)
            self.tip_function_src = vtk.vtkParametricFunctionSource()
            self.tip_function_src.SetParametricFunction(self.tip_spline)
            self.tip_function_src.SetUResolution(self.tip_pts.GetNumberOfPoints())
            self.tip_function_src.Update()
            # Radius interpolation
            self.tip_radius_interp = vtk.vtkTupleInterpolator()
            self.tip_radius_interp.SetInterpolationTypeToLinear()
            self.tip_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.tip_radius = vtk.vtkDoubleArray()
            N_spline = self.tip_function_src.GetOutput().GetNumberOfPoints()
            self.tip_radius.SetNumberOfTuples(N_spline)
            self.tip_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.tip_radius.SetTuple1(i, t)
            self.tubePolyData_tip = vtk.vtkPolyData()
            self.tubePolyData_tip = self.tip_function_src.GetOutput()
            self.tubePolyData_tip.GetPointData().AddArray(self.tip_radius)
            self.tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_tip = vtk.vtkTubeFilter()
            self.tuber_tip.SetInputData(self.tubePolyData_tip)
            self.tuber_tip.SetNumberOfSides(6)
            self.tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_tip)
            lineMapper.SetScalarRange(self.tubePolyData_tip.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_tip.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            #self.tubeActor_tip = vtk.vtkActor()
            self.tubeActor_tip.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_tip.SetMapper(tubeMapper)
        else:
            self.tip_pts = vtk.vtkPoints()
            self.tip_pts.SetDataTypeToFloat()
            # add dummy points:
            self.tip_pts.InsertPoint(0,(0.0,0.0,0.0))
            self.tip_pts.InsertPoint(1,(0.0,0.0,0.01))
            # root spline
            self.tip_spline = vtk.vtkParametricSpline()
            self.tip_spline.SetPoints(self.tip_pts)
            self.tip_function_src = vtk.vtkParametricFunctionSource()
            self.tip_function_src.SetParametricFunction(self.tip_spline)
            self.tip_function_src.SetUResolution(self.tip_pts.GetNumberOfPoints())
            self.tip_function_src.Update()
            # Radius interpolation
            self.tip_radius_interp = vtk.vtkTupleInterpolator()
            self.tip_radius_interp.SetInterpolationTypeToLinear()
            self.tip_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.tip_radius = vtk.vtkDoubleArray()
            N_spline = self.tip_function_src.GetOutput().GetNumberOfPoints()
            self.tip_radius.SetNumberOfTuples(N_spline)
            self.tip_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.tip_radius.SetTuple1(i, t)
            self.tubePolyData_tip = vtk.vtkPolyData()
            self.tubePolyData_tip = self.tip_function_src.GetOutput()
            self.tubePolyData_tip.GetPointData().AddArray(self.tip_radius)
            self.tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_tip = vtk.vtkTubeFilter()
            self.tuber_tip.SetInputData(self.tubePolyData_tip)
            self.tuber_tip.SetNumberOfSides(6)
            self.tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_tip)
            lineMapper.SetScalarRange(self.tubePolyData_tip.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_tip.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            self.tubeActor_tip = vtk.vtkActor()
            self.tubeActor_tip.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_tip.SetMapper(tubeMapper)

    def add_tip_trace(self,renderer):
        renderer.AddActor(self.tubeActor_tip)

    def remove_tip_trace(self,renderer):
        renderer.RemoveActor(self.tubeActor_tip)

    def set_root_axes(self):
        # Add axes:
        self.root_axes = vtk.vtkAxesActor()
        #self.root_axes.SetTotalLength(1.0,1.0,1.0)
        self.root_axes.SetTotalLength(1.8,1.8,1.8)
        self.root_axes.SetXAxisLabelText('')
        self.root_axes.SetYAxisLabelText('')
        self.root_axes.SetZAxisLabelText('')
        self.root_axes.SetShaftTypeToCylinder()
        self.root_axes.SetCylinderRadius(0.005)
        self.root_axes.SetConeRadius(0.1)

    def add_actors(self,renderer):
        renderer.AddActor(self.tubeActor_L0)
        renderer.AddActor(self.tubeActor_L1)
        renderer.AddActor(self.tubeActor_L2)
        renderer.AddActor(self.tubeActor_L3)
        renderer.AddActor(self.tubeActor_L4)
        renderer.AddActor(self.tubeActor_L5)
        renderer.AddActor(self.tubeActor_C1)
        renderer.AddActor(self.tubeActor_C2)
        renderer.AddActor(self.tubeActor_C3)
        renderer.AddActor(self.tubeActor_A)
        renderer.AddActor(self.tubeActor_P)
        renderer.AddActor(self.membrane_Actor_0)
        renderer.AddActor(self.membrane_Actor_1)
        renderer.AddActor(self.membrane_Actor_2)
        renderer.AddActor(self.membrane_Actor_3)
        if self.root_trace_on:
            renderer.AddActor(self.tubeActor_root)
        if self.tip_trace_on:
            renderer.AddActor(self.tubeActor_tip)
        if self.root_axes_on:
            renderer.AddActor(self.root_axes)

    def transform_wing(self,s_in):
        q_norm = np.sqrt(pow(s_in[0],2)+pow(s_in[1],2)+pow(s_in[2],2)+pow(s_in[3],2))
        q0 = s_in[0]/q_norm
        q1 = s_in[1]/q_norm
        q2 = s_in[2]/q_norm
        q3 = s_in[3]/q_norm
        tx = s_in[4]
        ty = s_in[5]
        tz = s_in[6]
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2, tx],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1, ty],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        b_angle = s_in[7]
        M_0 = self.convert_2_vtkMat(M)
        M1 = self.M_axis_1(M,b_angle/3.0)
        M_1 = self.convert_2_vtkMat(M1)
        M2 = self.M_axis_2(M1,b_angle/3.0)
        M_2 = self.convert_2_vtkMat(M2)
        M3 = self.M_axis_3(M2,b_angle/3.0)
        M_3 = self.convert_2_vtkMat(M3)
        self.tubeActor_L0.SetUserMatrix(M_0)
        self.tubeActor_L0.Modified()
        self.tubeActor_L1.SetUserMatrix(M_0)
        self.tubeActor_L1.Modified()
        self.tubeActor_L2.SetUserMatrix(M_0)
        self.tubeActor_L2.Modified()
        self.tubeActor_L3.SetUserMatrix(M_0)
        self.tubeActor_L3.Modified()
        self.tubeActor_C1.SetUserMatrix(M_1)
        self.tubeActor_C1.Modified()
        self.tubeActor_L4.SetUserMatrix(M_2)
        self.tubeActor_L4.Modified()
        self.tubeActor_C2.SetUserMatrix(M_2)
        self.tubeActor_C2.Modified()
        self.tubeActor_L5.SetUserMatrix(M_2)
        self.tubeActor_L5.Modified()
        self.tubeActor_C3.SetUserMatrix(M_3)
        self.tubeActor_C3.Modified()
        self.tubeActor_A.SetUserMatrix(M_1)
        self.tubeActor_A.Modified()
        self.tubeActor_P.SetUserMatrix(M_2)
        self.tubeActor_P.Modified()
        self.membrane_Actor_0.SetUserMatrix(M_0)
        self.membrane_Actor_0.Modified()
        self.membrane_Actor_1.SetUserMatrix(M_1)
        self.membrane_Actor_1.Modified()
        self.membrane_Actor_2.SetUserMatrix(M_2)
        self.membrane_Actor_2.Modified()
        self.membrane_Actor_3.SetUserMatrix(M_3)
        self.membrane_Actor_3.Modified()
        #self.renWin.Render()
        scale_arr = np.array([[self.scale_now],[self.scale_now],[self.scale_now],[1.0]])
        key0_pts = np.dot(M,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[0,1,2,8],:])))
        key1_pts = np.dot(M1,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[3,7,9],:])))
        key2_pts = np.dot(M2,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[4,6],:])))
        key3_pts = np.dot(M3,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[5,4],:])))
        self.key_points = np.zeros((3,10))
        self.key_points[:,0] = key0_pts[0:3,0] # keypoint 0
        self.key_points[:,1] = key0_pts[0:3,1] # keypoint 1
        self.key_points[:,2] = key0_pts[0:3,2] # keypoint 2
        self.key_points[:,3] = key1_pts[0:3,0] # keypoint 3
        self.key_points[:,4] = key2_pts[0:3,0] # keypoint 4
        self.key_points[:,5] = key3_pts[0:3,0] # keypoint 5
        self.key_points[:,6] = key2_pts[0:3,1] # keypoint 6
        self.key_points[:,7] = key1_pts[0:3,1] # keypoint 7
        self.key_points[:,8] = key0_pts[0:3,3] # keypoint 8
        self.key_points[:,9] = key1_pts[0:3,2] # keypoint 9
        # Update root and wing trace:
        self.root_pts_list.append(np.array([tx,ty,tz]))
        self.root_trace(self.root_pts_list,100000)
        self.tip_pts_list.append(self.key_points[:,2])
        self.tip_trace(self.tip_pts_list,100000)
        # Update root trace:
        root_transform = vtk.vtkTransform()
        root_transform.Translate(tx,ty,tz)
        self.root_axes.SetUserTransform(root_transform);
        self.root_axes.Modified()
        return M, M1, M2, M3

    def clear_root_tip_pts(self,root_pts_in,tip_pts_in):
        self.root_pts_list = [root_pts_in,root_pts_in]
        self.tip_pts_list = [tip_pts_in,tip_pts_in]
        self.root_trace(self.root_pts_list,100000)
        self.tip_trace(self.tip_pts_list,100000)
    
    def convert_2_vtkMat(self,M):
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
        return M_vtk

    def M_axis_1(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.0
        q2 = 1.0*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        return M_out

    def M_axis_2(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = -0.05959*np.sin(b_angle/2.0)
        q2 = 0.99822*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        TA = np.dot(R_out,np.array([-0.0867,0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        return M_out

    def M_axis_3(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = -0.36186*np.sin(b_angle/2.0)
        q2 = 0.93223*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        TA = np.dot(R_out,np.array([-0.0867,0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        return M_out

    def scale_wing(self,scale_in):
        self.scale_now = scale_in
        self.tubeActor_L0.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L0.Modified()
        self.tubeActor_L1.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L1.Modified()
        self.tubeActor_L2.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L2.Modified()
        self.tubeActor_L3.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L3.Modified()
        self.tubeActor_L4.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L4.Modified()
        self.tubeActor_L5.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L5.Modified()
        self.tubeActor_C1.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_C1.Modified()
        self.tubeActor_C2.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_C2.Modified()
        self.tubeActor_C3.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_C3.Modified()
        self.tubeActor_A.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_A.Modified()
        self.tubeActor_P.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_P.Modified()
        #self.membrane_Actor.SetScale(scale_in,scale_in,scale_in)
        #self.membrane_Actor.Modified()
        self.membrane_Actor_0.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_0.Modified()
        self.membrane_Actor_1.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_1.Modified()
        self.membrane_Actor_2.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_2.Modified()
        self.membrane_Actor_3.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_3.Modified()

    def return_polydata_mem0(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_0)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def return_polydata_mem1(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_1)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def return_polydata_mem2(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_2)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def return_polydata_mem3(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_3)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def write_stl(self,save_dir,scale_in):
        transform = vtk.vtkTransform()
        transform.Scale(scale_in,scale_in,scale_in)
        stlWriter = vtk.vtkSTLWriter()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_0)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_0_L.stl'))
        stlWriter.Write()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_1)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_1_L.stl'))
        stlWriter.Write()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_2)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_2_L.stl'))
        stlWriter.Write()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_3)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_3_L.stl'))
        stlWriter.Write()

    def set_Color(self,color_vec):
        self.vein_clr = color_vec[0]
        self.mem_clr = color_vec[1]
        self.mem_opacity = color_vec[2]
        self.tubeActor_L0.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L0.Modified()
        self.tubeActor_L1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L1.Modified()
        self.tubeActor_L2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L2.Modified()
        self.tubeActor_L3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L3.Modified()
        self.tubeActor_L4.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L4.Modified()
        self.tubeActor_L5.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L5.Modified()
        self.tubeActor_C1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C1.Modified()
        self.tubeActor_C2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C2.Modified()
        self.tubeActor_C3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C3.Modified()
        self.tubeActor_A.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_A.Modified()
        self.tubeActor_P.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_P.Modified()
        self.membrane_Actor_0.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_0.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_0.ForceTranslucentOn()
        self.membrane_Actor_0.Modified()
        self.membrane_Actor_1.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_1.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_1.ForceTranslucentOn()
        self.membrane_Actor_1.Modified()
        self.membrane_Actor_2.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_2.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_2.ForceTranslucentOn()
        self.membrane_Actor_2.Modified()
        self.membrane_Actor_3.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_3.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_3.ForceTranslucentOn()
        self.membrane_Actor_3.Modified()

    def visualize(self):
        self.ren = vtk.vtkRenderer()
        self.ren.SetUseDepthPeeling(True)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.ren.AddActor(self.tubeActor_L0)
        self.ren.AddActor(self.tubeActor_L1)
        self.ren.AddActor(self.tubeActor_L2)
        self.ren.AddActor(self.tubeActor_L3)
        self.ren.AddActor(self.tubeActor_L4)
        self.ren.AddActor(self.tubeActor_L5)
        self.ren.AddActor(self.tubeActor_C1)
        self.ren.AddActor(self.tubeActor_C2)
        self.ren.AddActor(self.tubeActor_C3)
        self.ren.AddActor(self.tubeActor_A)
        self.ren.AddActor(self.tubeActor_P)
        #self.ren.AddActor(self.membrane_Actor)
        self.ren.AddActor(self.membrane_Actor_0)
        self.ren.AddActor(self.membrane_Actor_1)
        self.ren.AddActor(self.membrane_Actor_2)
        self.ren.AddActor(self.membrane_Actor_3)
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(1200,1200)
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def plotKeyPoints(self,renderer,y_in,clr):
        # plot keypoints in 3D
        N_pts = 10
        for i in range(N_pts):
            key_sphere = vtk.vtkSphereSource()
            key_sphere.SetCenter(y_in[i*3],y_in[i*3+1],y_in[i*3+2])
            key_sphere.SetRadius(0.05)
            key_mapper = vtk.vtkPolyDataMapper()
            key_mapper.SetInputConnection(key_sphere.GetOutputPort())
            key_actor = vtk.vtkActor()
            key_actor.SetMapper(key_mapper)
            key_actor.GetProperty().SetColor(clr[0],clr[1],clr[2])
            renderer.AddActor(key_actor)

# Construct a model of a wing:
class WingModel_R():

    def __init__(self):
        self.vein_clr = (0.01,0.01,0.01)
        self.mem_clr = (0.01,0.01,0.01)
        self.mem_opacity = 0.3
        self.L0_vein()
        self.L1_vein()
        self.L2_vein()
        self.L3_vein()
        self.L4_vein()
        self.L5_vein()
        self.C1_vein()
        self.C2_vein()
        self.C3_vein()
        self.A_vein()
        self.P_vein()
        self.membrane_0()
        self.membrane_1()
        self.membrane_2()
        self.membrane_3()
        self.wing_key_pts = np.array([
            [0.2313, -0.5711, 0.0, 1.0],
            [0.3253, -2.3205, 0.0, 1.0],
            [0.0, -2.6241, 0.0, 1.0],
            [-0.2386, -2.5591, 0.0, 1.0],
            [-0.7012, -1.5976, 0.0, 1.0],
            [-0.7880, -0.8892, 0.0, 1.0],
            [-0.4048, -1.2578, 0.0, 1.0],
            [-0.1952, -1.2868, 0.0, 1.0],
            [0.0072, -0.7157, 0.0, 1.0],
            [-0.0867, -0.0145, 0.0, 1.0]])
        self.root_pts_list = []
        self.root_pts_list.append(np.zeros(3))
        self.root_trace(self.root_pts_list,500)
        self.tip_pts_list = []
        self.tip_pts_list.append(np.zeros(3))
        self.tip_trace(self.tip_pts_list,500)
        # Root axes:
        self.set_root_axes()
        # booleans:
        self.root_trace_on = False
        self.tip_trace_on = True
        self.root_axes_on = False

    def L0_vein(self):
        # L0 vein:
        self.L0_vein_pts = vtk.vtkPoints()
        self.L0_vein_pts.SetDataTypeToFloat()
        self.L0_vein_pts.InsertPoint(0,(0.0578,-0.0145,0.0))
        self.L0_vein_pts.InsertPoint(1,(0.1301,-0.0940,0.0))
        self.L0_vein_pts.InsertPoint(2,(0.1735,-0.1663,0.0))
        self.L0_vein_pts.InsertPoint(3,(0.2024,-0.2386,0.0))
        self.L0_vein_pts.InsertPoint(4,(0.2241,-0.3108,0.0))
        self.L0_vein_pts.InsertPoint(5,(0.2313,-0.3831,0.0))
        self.L0_vein_pts.InsertPoint(6,(0.2458,-0.4554,0.0))
        self.L0_vein_pts.InsertPoint(7,(0.2458,-0.5277,0.0))
        self.L0_vein_pts.InsertPoint(8,(0.2313,-0.5711,0.0))
        # Fit a spline to the points
        self.L0_spline = vtk.vtkParametricSpline()
        self.L0_spline.SetPoints(self.L0_vein_pts)
        self.L0_function_src = vtk.vtkParametricFunctionSource()
        self.L0_function_src.SetParametricFunction(self.L0_spline)
        self.L0_function_src.SetUResolution(self.L0_vein_pts.GetNumberOfPoints())
        self.L0_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L0_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L0_function_src.GetOutput().GetNumberOfPoints()
        self.L0_tube_radius.SetNumberOfTuples(N_spline)
        self.L0_tube_radius.SetName("TubeRadius")
        tMin = 0.02168
        tMax = 0.02602
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            #r = 1.0
            self.L0_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L0 = vtk.vtkPolyData()
        self.tubePolyData_L0 = self.L0_function_src.GetOutput()
        self.tubePolyData_L0.GetPointData().AddArray(self.L0_tube_radius)
        self.tubePolyData_L0.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L0 = vtk.vtkTubeFilter();
        self.tuber_L0.SetInputData(self.tubePolyData_L0);
        self.tuber_L0.SetNumberOfSides(6);
        self.tuber_L0.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        self.tuber_L0.Update()
        tuber_copy = self.tuber_L0.GetOutput()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L0)
        lineMapper.SetScalarRange(self.tubePolyData_L0.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L0.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L0.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L0 = vtk.vtkActor()
        self.tubeActor_L0.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L0.SetMapper(tubeMapper)

    def L1_vein(self):
        # L1 vein:
        self.L1_vein_pts = vtk.vtkPoints()
        self.L1_vein_pts.SetDataTypeToFloat()
        self.L1_vein_pts.InsertPoint(0,(0.0578,-0.0145,0.0))
        self.L1_vein_pts.InsertPoint(1,(0.0361,-0.0940,0.0))
        self.L1_vein_pts.InsertPoint(2,(0.0361,-0.1663,0.0))
        self.L1_vein_pts.InsertPoint(3,(0.0506,-0.2386,0.0))
        self.L1_vein_pts.InsertPoint(4,(0.0651,-0.3109,0.0))
        self.L1_vein_pts.InsertPoint(5,(0.1012,-0.3831,0.0))
        self.L1_vein_pts.InsertPoint(6,(0.1374,-0.4554,0.0))
        self.L1_vein_pts.InsertPoint(7,(0.1952,-0.5277,0.0))
        self.L1_vein_pts.InsertPoint(8,(0.2386,-0.5639,0.0))
        self.L1_vein_pts.InsertPoint(9,(0.2458,-0.6000,0.0))
        self.L1_vein_pts.InsertPoint(10,(0.2747,-0.6723,0.0))
        self.L1_vein_pts.InsertPoint(11,(0.3036,-0.7446,0.0))
        self.L1_vein_pts.InsertPoint(12,(0.3181,-0.8169,0.0))
        self.L1_vein_pts.InsertPoint(13,(0.3398,-0.8892,0.0))
        self.L1_vein_pts.InsertPoint(14,(0.3542,-0.9615,0.0))
        self.L1_vein_pts.InsertPoint(15,(0.3687,-1.0337,0.0))
        self.L1_vein_pts.InsertPoint(16,(0.3831,-1.1060,0.0))
        self.L1_vein_pts.InsertPoint(17,(0.3904,-1.1783,0.0))
        self.L1_vein_pts.InsertPoint(18,(0.3976,-1.2506,0.0))
        self.L1_vein_pts.InsertPoint(19,(0.4048,-1.3229,0.0))
        self.L1_vein_pts.InsertPoint(20,(0.4121,-1.3952,0.0))
        self.L1_vein_pts.InsertPoint(21,(0.4193,-1.4675,0.0))
        self.L1_vein_pts.InsertPoint(22,(0.4193,-1.5400,0.0))
        self.L1_vein_pts.InsertPoint(23,(0.4265,-1.6121,0.0))
        self.L1_vein_pts.InsertPoint(24,(0.4265,-1.6844,0.0))
        self.L1_vein_pts.InsertPoint(25,(0.4337,-1.7566,0.0))
        self.L1_vein_pts.InsertPoint(26,(0.4265,-1.8289,0.0))
        self.L1_vein_pts.InsertPoint(27,(0.4193,-1.9012,0.0))
        self.L1_vein_pts.InsertPoint(28,(0.4121,-1.9735,0.0))
        self.L1_vein_pts.InsertPoint(29,(0.4048,-2.0458,0.0))
        self.L1_vein_pts.InsertPoint(30,(0.3904,-2.1181,0.0))
        self.L1_vein_pts.InsertPoint(31,(0.3759,-2.1904,0.0))
        self.L1_vein_pts.InsertPoint(32,(0.3542,-2.2627,0.0))
        self.L1_vein_pts.InsertPoint(33,(0.3253,-2.3350,0.0))
        self.L1_vein_pts.InsertPoint(34,(0.2819,-2.4073,0.0))
        self.L1_vein_pts.InsertPoint(35,(0.2241,-2.4795,0.0))
        self.L1_vein_pts.InsertPoint(36,(0.1374,-2.5518,0.0))
        self.L1_vein_pts.InsertPoint(37,(0.0867,-2.5880,0.0))
        self.L1_vein_pts.InsertPoint(38,(0.0506,-2.6097,0.0))
        self.L1_vein_pts.InsertPoint(39,(0.0,-2.6241,0.0))
        # Fit a spline to the points
        self.L1_spline = vtk.vtkParametricSpline()
        self.L1_spline.SetPoints(self.L1_vein_pts)
        self.L1_function_src = vtk.vtkParametricFunctionSource()
        self.L1_function_src.SetParametricFunction(self.L1_spline)
        self.L1_function_src.SetUResolution(self.L1_vein_pts.GetNumberOfPoints())
        self.L1_function_src.Update()
        # Tube radius
        self.L1_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L1_function_src.GetOutput().GetNumberOfPoints()
        self.L1_tube_radius.SetNumberOfTuples(N_spline)
        self.L1_tube_radius.SetName("TubeRadius")
        tMin = 0.03614
        tMax = 0.01444
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L1_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L1 = vtk.vtkPolyData()
        self.tubePolyData_L1 = self.L1_function_src.GetOutput()
        self.tubePolyData_L1.GetPointData().AddArray(self.L1_tube_radius)
        self.tubePolyData_L1.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L1 = vtk.vtkTubeFilter();
        self.tuber_L1.SetInputData(self.tubePolyData_L1);
        self.tuber_L1.SetNumberOfSides(6);
        self.tuber_L1.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L1)
        lineMapper.SetScalarRange(self.tubePolyData_L1.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L1.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L1.GetScalarRange())
        # Line Actor
        lineActor_L1 = vtk.vtkActor()
        lineActor_L1.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L1 = vtk.vtkActor()
        self.tubeActor_L1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L1.SetMapper(tubeMapper)

    def L2_vein(self):
        # L2 vein:
        self.L2_vein_pts = vtk.vtkPoints()
        self.L2_vein_pts.SetDataTypeToFloat()
        self.L2_vein_pts.InsertPoint(0,(0.0361,-0.0940,0.0))
        self.L2_vein_pts.InsertPoint(1,(-0.0145,-0.1663,0.0))
        self.L2_vein_pts.InsertPoint(2,(-0.0145,-0.2241,0.0))
        self.L2_vein_pts.InsertPoint(3,(0.0145,-0.3109,0.0))
        self.L2_vein_pts.InsertPoint(4,(0.0506,-0.3831,0.0))
        self.L2_vein_pts.InsertPoint(5,(0.0723,-0.4554,0.0))
        self.L2_vein_pts.InsertPoint(6,(0.0795,-0.5277,0.0))
        self.L2_vein_pts.InsertPoint(7,(0.0867,-0.6000,0.0))
        self.L2_vein_pts.InsertPoint(8,(0.1084,-0.6723,0.0))
        self.L2_vein_pts.InsertPoint(9,(0.1229,-0.7446,0.0))
        self.L2_vein_pts.InsertPoint(10,(0.1374,-0.8169,0.0))
        self.L2_vein_pts.InsertPoint(11,(0.1518,-0.8892,0.0))
        self.L2_vein_pts.InsertPoint(12,(0.1735,-0.9615,0.0))
        self.L2_vein_pts.InsertPoint(13,(0.1880,-1.0337,0.0))
        self.L2_vein_pts.InsertPoint(14,(0.2096,-1.1060,0.0))
        self.L2_vein_pts.InsertPoint(15,(0.2241,-1.1783,0.0))
        self.L2_vein_pts.InsertPoint(16,(0.2386,-1.2506,0.0))
        self.L2_vein_pts.InsertPoint(17,(0.2458,-1.3229,0.0))
        self.L2_vein_pts.InsertPoint(18,(0.2602,-1.3952,0.0))
        self.L2_vein_pts.InsertPoint(19,(0.2675,-1.4675,0.0))
        self.L2_vein_pts.InsertPoint(20,(0.2747,-1.5398,0.0))
        self.L2_vein_pts.InsertPoint(21,(0.2747,-1.6121,0.0))
        self.L2_vein_pts.InsertPoint(22,(0.2747,-1.6844,0.0))
        self.L2_vein_pts.InsertPoint(23,(0.2747,-1.7566,0.0))
        self.L2_vein_pts.InsertPoint(24,(0.2819,-1.8289,0.0))
        self.L2_vein_pts.InsertPoint(25,(0.2819,-1.9012,0.0))
        self.L2_vein_pts.InsertPoint(26,(0.2819,-1.9735,0.0))
        self.L2_vein_pts.InsertPoint(27,(0.2819,-2.0458,0.0))
        self.L2_vein_pts.InsertPoint(28,(0.2892,-2.1181,0.0))
        self.L2_vein_pts.InsertPoint(29,(0.2892,-2.1904,0.0))
        self.L2_vein_pts.InsertPoint(30,(0.3036,-2.2627,0.0))
        self.L2_vein_pts.InsertPoint(31,(0.3253,-2.3205,0.0))
        # Fit a spline to the points
        self.L2_spline = vtk.vtkParametricSpline()
        self.L2_spline.SetPoints(self.L2_vein_pts)
        self.L2_function_src = vtk.vtkParametricFunctionSource()
        self.L2_function_src.SetParametricFunction(self.L2_spline)
        self.L2_function_src.SetUResolution(self.L2_vein_pts.GetNumberOfPoints())
        self.L2_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L2_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L2_function_src.GetOutput().GetNumberOfPoints()
        self.L2_tube_radius.SetNumberOfTuples(N_spline)
        self.L2_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.01156
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L2_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L2 = vtk.vtkPolyData()
        self.tubePolyData_L2 = self.L2_function_src.GetOutput()
        self.tubePolyData_L2.GetPointData().AddArray(self.L2_tube_radius)
        self.tubePolyData_L2.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L2 = vtk.vtkTubeFilter();
        self.tuber_L2.SetInputData(self.tubePolyData_L2);
        self.tuber_L2.SetNumberOfSides(6);
        self.tuber_L2.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L2)
        lineMapper.SetScalarRange(self.tubePolyData_L2.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L2.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetScalarRange(self.tubePolyData_L2.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L2 = vtk.vtkActor()
        self.tubeActor_L2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L2.SetMapper(tubeMapper)

    def L3_vein(self):
        # L3 vein:
        self.L3_vein_pts = vtk.vtkPoints()
        self.L3_vein_pts.SetDataTypeToFloat()
        self.L3_vein_pts.InsertPoint(0,(-0.0145,-0.2241,0.0))
        self.L3_vein_pts.InsertPoint(1,(0.0,-0.3108,0.0))
        self.L3_vein_pts.InsertPoint(2,(0.0145,-0.3831,0.0))
        self.L3_vein_pts.InsertPoint(3,(0.0145,-0.4554,0.0))
        self.L3_vein_pts.InsertPoint(4,(0.0072,-0.5277,0.0))
        self.L3_vein_pts.InsertPoint(5,(0.0072,-0.6000,0.0))
        self.L3_vein_pts.InsertPoint(6,(0.0,-0.6723,0.0))
        self.L3_vein_pts.InsertPoint(7,(0.0072,-0.7157,0.0))
        self.L3_vein_pts.InsertPoint(8,(0.0145,-0.7446,0.0))
        self.L3_vein_pts.InsertPoint(9,(0.0217,-0.8169,0.0))
        self.L3_vein_pts.InsertPoint(10,(0.0361,-0.8892,0.0))
        self.L3_vein_pts.InsertPoint(11,(0.0434,-0.9615,0.0))
        self.L3_vein_pts.InsertPoint(12,(0.0506,-1.0337,0.0))
        self.L3_vein_pts.InsertPoint(13,(0.0578,-1.1060,0.0))
        self.L3_vein_pts.InsertPoint(14,(0.0578,-1.1783,0.0))
        self.L3_vein_pts.InsertPoint(15,(0.0578,-1.2506,0.0))
        self.L3_vein_pts.InsertPoint(16,(0.0578,-1.3229,0.0))
        self.L3_vein_pts.InsertPoint(17,(0.0578,-1.3952,0.0))
        self.L3_vein_pts.InsertPoint(18,(0.0651,-1.4675,0.0))
        self.L3_vein_pts.InsertPoint(19,(0.0651,-1.5398,0.0))
        self.L3_vein_pts.InsertPoint(20,(0.0723,-1.6121,0.0))
        self.L3_vein_pts.InsertPoint(21,(0.0723,-1.6844,0.0))
        self.L3_vein_pts.InsertPoint(22,(0.0723,-1.7566,0.0))
        self.L3_vein_pts.InsertPoint(23,(0.0723,-1.8289,0.0))
        self.L3_vein_pts.InsertPoint(24,(0.0651,-1.9012,0.0))
        self.L3_vein_pts.InsertPoint(25,(0.0651,-1.9735,0.0))
        self.L3_vein_pts.InsertPoint(26,(0.0578,-2.0458,0.0))
        self.L3_vein_pts.InsertPoint(27,(0.0578,-2.1181,0.0))
        self.L3_vein_pts.InsertPoint(28,(0.0506,-2.1904,0.0))
        self.L3_vein_pts.InsertPoint(29,(0.0506,-2.2627,0.0))
        self.L3_vein_pts.InsertPoint(30,(0.0361,-2.3350,0.0))
        self.L3_vein_pts.InsertPoint(31,(0.0217,-2.4073,0.0))
        self.L3_vein_pts.InsertPoint(32,(0.0145,-2.4795,0.0))
        self.L3_vein_pts.InsertPoint(33,(0.0072,-2.5518,0.0))
        self.L3_vein_pts.InsertPoint(34,(0.0,-2.6241,0.0))
        # Fit a spline to the points
        self.L3_spline = vtk.vtkParametricSpline()
        self.L3_spline.SetPoints(self.L3_vein_pts)
        self.L3_function_src = vtk.vtkParametricFunctionSource()
        self.L3_function_src.SetParametricFunction(self.L3_spline)
        self.L3_function_src.SetUResolution(self.L3_vein_pts.GetNumberOfPoints())
        self.L3_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L3_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L3_function_src.GetOutput().GetNumberOfPoints()
        self.L3_tube_radius.SetNumberOfTuples(N_spline)
        self.L3_tube_radius.SetName("TubeRadius")
        tMin = 0.01302
        tMax = 0.01302
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L3_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L3 = vtk.vtkPolyData()
        self.tubePolyData_L3 = self.L3_function_src.GetOutput()
        self.tubePolyData_L3.GetPointData().AddArray(self.L3_tube_radius)
        self.tubePolyData_L3.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L3 = vtk.vtkTubeFilter();
        self.tuber_L3.SetInputData(self.tubePolyData_L3);
        self.tuber_L3.SetNumberOfSides(6);
        self.tuber_L3.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L3)
        lineMapper.SetScalarRange(self.tubePolyData_L3.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L3.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L3.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L3 = vtk.vtkActor()
        self.tubeActor_L3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L3.SetMapper(tubeMapper)

    def L4_vein(self):
        # L4 vein:
        self.L4_vein_pts = vtk.vtkPoints()
        self.L4_vein_pts.SetDataTypeToFloat()
        self.L4_vein_pts.InsertPoint(0,(-0.0867,-0.0145,0.0))
        self.L4_vein_pts.InsertPoint(1,(-0.0651,-0.0940,0.0))
        self.L4_vein_pts.InsertPoint(2,(-0.0506,-0.1663,0.0))
        self.L4_vein_pts.InsertPoint(3,(-0.0723,-0.2386,0.0))
        self.L4_vein_pts.InsertPoint(4,(-0.0795,-0.3108,0.0))
        self.L4_vein_pts.InsertPoint(5,(-0.0795,-0.3831,0.0))
        self.L4_vein_pts.InsertPoint(6,(-0.0867,-0.4554,0.0))
        self.L4_vein_pts.InsertPoint(7,(-0.0867,-0.5277,0.0))
        self.L4_vein_pts.InsertPoint(8,(-0.0940,-0.6000,0.0))
        self.L4_vein_pts.InsertPoint(9,(-0.0867,-0.6723,0.0))
        self.L4_vein_pts.InsertPoint(10,(-0.0795,-0.6940,0.0))
        self.L4_vein_pts.InsertPoint(11,(-0.1012,-0.7446,0.0))
        self.L4_vein_pts.InsertPoint(12,(-0.1084,-0.8169,0.0))
        self.L4_vein_pts.InsertPoint(13,(-0.1301,-0.8892,0.0))
        self.L4_vein_pts.InsertPoint(14,(-0.1446,-0.9615,0.0))
        self.L4_vein_pts.InsertPoint(15,(-0.1518,-1.0337,0.0))
        self.L4_vein_pts.InsertPoint(16,(-0.1663,-1.1060,0.0))
        self.L4_vein_pts.InsertPoint(17,(-0.1807,-1.1783,0.0))
        self.L4_vein_pts.InsertPoint(18,(-0.1880,-1.2506,0.0))
        self.L4_vein_pts.InsertPoint(19,(-0.2024,-1.2868,0.0))
        self.L4_vein_pts.InsertPoint(20,(-0.1952,-1.3229,0.0))
        self.L4_vein_pts.InsertPoint(21,(-0.1880,-1.3952,0.0))
        self.L4_vein_pts.InsertPoint(22,(-0.1880,-1.4675,0.0))
        self.L4_vein_pts.InsertPoint(23,(-0.1880,-1.5398,0.0))
        self.L4_vein_pts.InsertPoint(24,(-0.1880,-1.6121,0.0))
        self.L4_vein_pts.InsertPoint(25,(-0.1880,-1.6844,0.0))
        self.L4_vein_pts.InsertPoint(26,(-0.1880,-1.7566,0.0))
        self.L4_vein_pts.InsertPoint(27,(-0.1880,-1.8289,0.0))
        self.L4_vein_pts.InsertPoint(28,(-0.1880,-1.9012,0.0))
        self.L4_vein_pts.InsertPoint(29,(-0.1952,-1.9735,0.0))
        self.L4_vein_pts.InsertPoint(30,(-0.1952,-2.0458,0.0))
        self.L4_vein_pts.InsertPoint(31,(-0.1952,-2.1181,0.0))
        self.L4_vein_pts.InsertPoint(32,(-0.1952,-2.1904,0.0))
        self.L4_vein_pts.InsertPoint(33,(-0.2024,-2.2627,0.0))
        self.L4_vein_pts.InsertPoint(34,(-0.2024,-2.3350,0.0))
        self.L4_vein_pts.InsertPoint(35,(-0.2096,-2.4073,0.0))
        self.L4_vein_pts.InsertPoint(36,(-0.2169,-2.4795,0.0))
        self.L4_vein_pts.InsertPoint(37,(-0.2386,-2.5591,0.0))
        # Fit a spline to the points
        self.L4_spline = vtk.vtkParametricSpline()
        self.L4_spline.SetPoints(self.L4_vein_pts)
        self.L4_function_src = vtk.vtkParametricFunctionSource()
        self.L4_function_src.SetParametricFunction(self.L4_spline)
        self.L4_function_src.SetUResolution(self.L4_vein_pts.GetNumberOfPoints())
        self.L4_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L4_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L4_function_src.GetOutput().GetNumberOfPoints()
        self.L4_tube_radius.SetNumberOfTuples(N_spline)
        self.L4_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.00867
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L4_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L4 = vtk.vtkPolyData()
        self.tubePolyData_L4 = self.L4_function_src.GetOutput()
        self.tubePolyData_L4.GetPointData().AddArray(self.L4_tube_radius)
        self.tubePolyData_L4.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L4 = vtk.vtkTubeFilter();
        self.tuber_L4.SetInputData(self.tubePolyData_L4);
        self.tuber_L4.SetNumberOfSides(6);
        self.tuber_L4.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L4)
        lineMapper.SetScalarRange(self.tubePolyData_L4.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L4.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L4.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L4 = vtk.vtkActor()
        self.tubeActor_L4.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L4.SetMapper(tubeMapper)

    def L5_vein(self):
        # L5 vein:
        self.L5_vein_pts = vtk.vtkPoints()
        self.L5_vein_pts.SetDataTypeToFloat()
        self.L5_vein_pts.InsertPoint(0,(-0.0867,-0.0145,0.0))
        self.L5_vein_pts.InsertPoint(1,(-0.1229,-0.0940,0.0))
        self.L5_vein_pts.InsertPoint(2,(-0.1446,-0.1663,0.0))
        self.L5_vein_pts.InsertPoint(3,(-0.1663,-0.2386,0.0))
        self.L5_vein_pts.InsertPoint(4,(-0.1807,-0.3108,0.0))
        self.L5_vein_pts.InsertPoint(5,(-0.1952,-0.3831,0.0))
        self.L5_vein_pts.InsertPoint(6,(-0.2024,-0.4554,0.0))
        self.L5_vein_pts.InsertPoint(7,(-0.2241,-0.5277,0.0))
        self.L5_vein_pts.InsertPoint(8,(-0.2386,-0.6000,0.0))
        self.L5_vein_pts.InsertPoint(9,(-0.2602,-0.6723,0.0))
        self.L5_vein_pts.InsertPoint(10,(-0.2747,-0.7446,0.0))
        self.L5_vein_pts.InsertPoint(11,(-0.2892,-0.8169,0.0))
        self.L5_vein_pts.InsertPoint(12,(-0.3108,-0.8892,0.0))
        self.L5_vein_pts.InsertPoint(13,(-0.3253,-0.9615,0.0))
        self.L5_vein_pts.InsertPoint(14,(-0.3470,-1.0337,0.0))
        self.L5_vein_pts.InsertPoint(15,(-0.3615,-1.1060,0.0))
        self.L5_vein_pts.InsertPoint(16,(-0.3904,-1.1783,0.0))
        self.L5_vein_pts.InsertPoint(17,(-0.4048,-1.2506,0.0))
        self.L5_vein_pts.InsertPoint(18,(-0.4410,-1.3229,0.0))
        self.L5_vein_pts.InsertPoint(19,(-0.4988,-1.4313,0.0))
        self.L5_vein_pts.InsertPoint(20,(-0.5494,-1.4675,0.0))
        self.L5_vein_pts.InsertPoint(21,(-0.6217,-1.5398,0.0))
        self.L5_vein_pts.InsertPoint(22,(-0.7012,-1.5976,0.0))
        # Fit a spline to the points
        self.L5_spline = vtk.vtkParametricSpline()
        self.L5_spline.SetPoints(self.L5_vein_pts)
        self.L5_function_src = vtk.vtkParametricFunctionSource()
        self.L5_function_src.SetParametricFunction(self.L5_spline)
        self.L5_function_src.SetUResolution(self.L5_vein_pts.GetNumberOfPoints())
        self.L5_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.L5_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.L5_function_src.GetOutput().GetNumberOfPoints()
        self.L5_tube_radius.SetNumberOfTuples(N_spline)
        self.L5_tube_radius.SetName("TubeRadius")
        tMin = 0.01734
        tMax = 0.00725
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.L5_tube_radius.SetTuple1(i, t)
        self.tubePolyData_L5 = vtk.vtkPolyData()
        self.tubePolyData_L5 = self.L5_function_src.GetOutput()
        self.tubePolyData_L5.GetPointData().AddArray(self.L5_tube_radius)
        self.tubePolyData_L5.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_L5 = vtk.vtkTubeFilter();
        self.tuber_L5.SetInputData(self.tubePolyData_L5);
        self.tuber_L5.SetNumberOfSides(6);
        self.tuber_L5.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_L5)
        lineMapper.SetScalarRange(self.tubePolyData_L5.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_L5.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_L5.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_L5 = vtk.vtkActor()
        self.tubeActor_L5.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L5.SetMapper(tubeMapper)

    def C1_vein(self):
        # C1 vein:
        self.C1_vein_pts = vtk.vtkPoints()
        self.C1_vein_pts.SetDataTypeToFloat()
        self.C1_vein_pts.InsertPoint(0,(0.0,-2.6241,0.0))
        self.C1_vein_pts.InsertPoint(1,(-0.0578,-2.6241,0.0))
        self.C1_vein_pts.InsertPoint(2,(-0.1374,-2.6097,0.0))
        self.C1_vein_pts.InsertPoint(3,(-0.1880,-2.5880,0.0))
        self.C1_vein_pts.InsertPoint(4,(-0.2386,-2.5518,0.0))
        # Fit a spline to the points
        self.C1_spline = vtk.vtkParametricSpline()
        self.C1_spline.SetPoints(self.C1_vein_pts)
        self.C1_function_src = vtk.vtkParametricFunctionSource()
        self.C1_function_src.SetParametricFunction(self.C1_spline)
        self.C1_function_src.SetUResolution(self.C1_vein_pts.GetNumberOfPoints())
        self.C1_function_src.Update()
        # Tube radius
        self.C1_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.C1_function_src.GetOutput().GetNumberOfPoints()
        self.C1_tube_radius.SetNumberOfTuples(N_spline)
        self.C1_tube_radius.SetName("TubeRadius")
        tMin = 0.01444
        tMax = 0.00868
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.C1_tube_radius.SetTuple1(i, t)
        self.tubePolyData_C1 = vtk.vtkPolyData()
        self.tubePolyData_C1 = self.C1_function_src.GetOutput()
        self.tubePolyData_C1.GetPointData().AddArray(self.C1_tube_radius)
        self.tubePolyData_C1.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_C1 = vtk.vtkTubeFilter();
        self.tuber_C1.SetInputData(self.tubePolyData_C1);
        self.tuber_C1.SetNumberOfSides(6);
        self.tuber_C1.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_C1)
        lineMapper.SetScalarRange(self.tubePolyData_C1.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_C1.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_C1 = vtk.vtkActor()
        self.tubeActor_C1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C1.SetMapper(tubeMapper)

    def C2_vein(self):
        # C2 vein:
        self.C2_vein_pts = vtk.vtkPoints()
        self.C2_vein_pts.SetDataTypeToFloat()
        self.C2_vein_pts.InsertPoint(0,(-0.2386,-2.5518,0.0))
        self.C2_vein_pts.InsertPoint(1,(-0.3181,-2.4795,0.0))
        self.C2_vein_pts.InsertPoint(2,(-0.3904,-2.4073,0.0))
        self.C2_vein_pts.InsertPoint(3,(-0.4410,-2.3350,0.0))
        self.C2_vein_pts.InsertPoint(4,(-0.4916,-2.2627,0.0))
        self.C2_vein_pts.InsertPoint(5,(-0.5277,-2.1904,0.0))
        self.C2_vein_pts.InsertPoint(6,(-0.5566,-2.1181,0.0))
        self.C2_vein_pts.InsertPoint(7,(-0.5928,-2.0458,0.0))
        self.C2_vein_pts.InsertPoint(8,(-0.6145,-1.9735,0.0))
        self.C2_vein_pts.InsertPoint(9,(-0.6362,-1.9012,0.0))
        self.C2_vein_pts.InsertPoint(10,(-0.6578,-1.8289,0.0))
        self.C2_vein_pts.InsertPoint(11,(-0.6795,-1.7566,0.0))
        self.C2_vein_pts.InsertPoint(12,(-0.7012,-1.6844,0.0))
        self.C2_vein_pts.InsertPoint(13,(-0.7012,-1.6121,0.0))
        self.C2_vein_pts.InsertPoint(14,(-0.7012,-1.5976,0.0))
        # Fit a spline to the points
        self.C2_spline = vtk.vtkParametricSpline()
        self.C2_spline.SetPoints(self.C2_vein_pts)
        self.C2_function_src = vtk.vtkParametricFunctionSource()
        self.C2_function_src.SetParametricFunction(self.C2_spline)
        self.C2_function_src.SetUResolution(self.C2_vein_pts.GetNumberOfPoints())
        self.C2_function_src.Update()
        # Tube radius
        self.C2_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.C2_function_src.GetOutput().GetNumberOfPoints()
        self.C2_tube_radius.SetNumberOfTuples(N_spline)
        self.C2_tube_radius.SetName("TubeRadius")
        tMin = 0.00868
        tMax = 0.00868
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.C2_tube_radius.SetTuple1(i, t)
        self.tubePolyData_C2 = vtk.vtkPolyData()
        self.tubePolyData_C2 = self.C2_function_src.GetOutput()
        self.tubePolyData_C2.GetPointData().AddArray(self.C2_tube_radius)
        self.tubePolyData_C2.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_C2 = vtk.vtkTubeFilter();
        self.tuber_C2.SetInputData(self.tubePolyData_C2);
        self.tuber_C2.SetNumberOfSides(6);
        self.tuber_C2.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_C2)
        lineMapper.SetScalarRange(self.tubePolyData_C2.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_C2.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_C2 = vtk.vtkActor()
        self.tubeActor_C2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C2.SetMapper(tubeMapper)

    def C3_vein(self):
        # C3 vein:
        self.C3_vein_pts = vtk.vtkPoints()
        self.C3_vein_pts.SetDataTypeToFloat()
        self.C3_vein_pts.InsertPoint(0,(-0.7012,-1.5976,0.0))
        self.C3_vein_pts.InsertPoint(1,(-0.7229,-1.5398,0.0))
        self.C3_vein_pts.InsertPoint(2,(-0.7446,-1.4675,0.0))
        self.C3_vein_pts.InsertPoint(3,(-0.7518,-1.3952,0.0))
        self.C3_vein_pts.InsertPoint(4,(-0.7663,-1.3229,0.0))
        self.C3_vein_pts.InsertPoint(5,(-0.7735,-1.2506,0.0))
        self.C3_vein_pts.InsertPoint(6,(-0.7807,-1.1783,0.0))
        self.C3_vein_pts.InsertPoint(7,(-0.7807,-1.1060,0.0))
        self.C3_vein_pts.InsertPoint(8,(-0.7807,-1.0337,0.0))
        self.C3_vein_pts.InsertPoint(9,(-0.7880,-0.9615,0.0))
        self.C3_vein_pts.InsertPoint(10,(-0.7880,-0.8892,0.0))
        self.C3_vein_pts.InsertPoint(11,(-0.7880,-0.8169,0.0))
        self.C3_vein_pts.InsertPoint(12,(-0.7807,-0.7446,0.0))
        self.C3_vein_pts.InsertPoint(13,(-0.7663,-0.6723,0.0))
        self.C3_vein_pts.InsertPoint(14,(-0.7590,-0.6000,0.0))
        self.C3_vein_pts.InsertPoint(15,(-0.7446,-0.5277,0.0))
        self.C3_vein_pts.InsertPoint(16,(-0.7229,-0.4554,0.0))
        self.C3_vein_pts.InsertPoint(17,(-0.6940,-0.3831,0.0))
        self.C3_vein_pts.InsertPoint(18,(-0.6434,-0.3108,0.0))
        self.C3_vein_pts.InsertPoint(19,(-0.5566,-0.2386,0.0))
        self.C3_vein_pts.InsertPoint(20,(-0.3831,-0.1663,0.0))
        self.C3_vein_pts.InsertPoint(21,(-0.2169,-0.0940,0.0))
        self.C3_vein_pts.InsertPoint(22,(-0.1157,-0.0217,0.0))
        self.C3_vein_pts.InsertPoint(23,(-0.0867,-0.0145,0.0))
        # Fit a spline to the point
        self.C3_spline = vtk.vtkParametricSpline()
        self.C3_spline.SetPoints(self.C3_vein_pts)
        self.C3_function_src = vtk.vtkParametricFunctionSource()
        self.C3_function_src.SetParametricFunction(self.C3_spline)
        self.C3_function_src.SetUResolution(self.C3_vein_pts.GetNumberOfPoints())
        self.C3_function_src.Update()
        # Tube radius
        self.C3_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.C3_function_src.GetOutput().GetNumberOfPoints()
        self.C3_tube_radius.SetNumberOfTuples(N_spline)
        self.C3_tube_radius.SetName("TubeRadius")
        tMin = 0.00868
        tMax = 0.00868
        for i in range(N_spline):
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.C3_tube_radius.SetTuple1(i, t)
        self.tubePolyData_C3 = vtk.vtkPolyData()
        self.tubePolyData_C3 = self.C3_function_src.GetOutput()
        self.tubePolyData_C3.GetPointData().AddArray(self.C3_tube_radius)
        self.tubePolyData_C3.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_C3 = vtk.vtkTubeFilter();
        self.tuber_C3.SetInputData(self.tubePolyData_C3);
        self.tuber_C3.SetNumberOfSides(6);
        self.tuber_C3.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_C3)
        lineMapper.SetScalarRange(self.tubePolyData_C3.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_C3.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_C3 = vtk.vtkActor()
        self.tubeActor_C3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C3.SetMapper(tubeMapper)

    def A_vein(self):
        # A vein:
        self.A_vein_pts = vtk.vtkPoints()
        self.A_vein_pts.SetDataTypeToFloat()
        self.A_vein_pts.InsertPoint(0,(0.0072,-0.7157,0.0))
        self.A_vein_pts.InsertPoint(1,(-0.0795,-0.6940,0.0))
        # Fit a spline to the points
        self.A_spline = vtk.vtkParametricSpline()
        self.A_spline.SetPoints(self.A_vein_pts)
        self.A_function_src = vtk.vtkParametricFunctionSource()
        self.A_function_src.SetParametricFunction(self.A_spline)
        self.A_function_src.SetUResolution(self.A_vein_pts.GetNumberOfPoints())
        self.A_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.A_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.A_function_src.GetOutput().GetNumberOfPoints()
        self.A_tube_radius.SetNumberOfTuples(N_spline)
        self.A_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.01156
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.A_tube_radius.SetTuple1(i, t)
        self.tubePolyData_A = vtk.vtkPolyData()
        self.tubePolyData_A = self.A_function_src.GetOutput()
        self.tubePolyData_A.GetPointData().AddArray(self.A_tube_radius)
        self.tubePolyData_A.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_A = vtk.vtkTubeFilter();
        self.tuber_A.SetInputData(self.tubePolyData_A);
        self.tuber_A.SetNumberOfSides(6);
        self.tuber_A.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_A)
        lineMapper.SetScalarRange(self.tubePolyData_A.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_A.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_A.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_A = vtk.vtkActor()
        self.tubeActor_A.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_A.SetMapper(tubeMapper)

    def P_vein(self):
        # P vein:
        self.P_vein_pts = vtk.vtkPoints()
        self.P_vein_pts.SetDataTypeToFloat()
        self.P_vein_pts.InsertPoint(0,(-0.1952,-1.2868,0.0))
        self.P_vein_pts.InsertPoint(1,(-0.3325,-1.2868,0.0))
        self.P_vein_pts.InsertPoint(2,(-0.4048,-1.2578,0.0))
        # Fit a spline to the points
        self.P_spline = vtk.vtkParametricSpline()
        self.P_spline.SetPoints(self.P_vein_pts)
        self.P_function_src = vtk.vtkParametricFunctionSource()
        self.P_function_src.SetParametricFunction(self.P_spline)
        self.P_function_src.SetUResolution(self.P_vein_pts.GetNumberOfPoints())
        self.P_function_src.Update()
        # Radius interpolation
        radius_interp = vtk.vtkTupleInterpolator()
        radius_interp.SetInterpolationTypeToLinear()
        radius_interp.SetNumberOfComponents(1)
        # Tube radius
        self.P_tube_radius = vtk.vtkDoubleArray()
        N_spline = self.P_function_src.GetOutput().GetNumberOfPoints()
        self.P_tube_radius.SetNumberOfTuples(N_spline)
        self.P_tube_radius.SetName("TubeRadius")
        tMin = 0.01156
        tMax = 0.01156
        #print('N_spline')
        #print(N_spline)
        for i in range(N_spline):
        #    #radius_interp.InterpolateTuple(tMax,tMin)
            t = (tMax-tMin)/(N_spline-1.0)*i+tMin
            self.P_tube_radius.SetTuple1(i, t)
        self.tubePolyData_P = vtk.vtkPolyData()
        self.tubePolyData_P = self.P_function_src.GetOutput()
        self.tubePolyData_P.GetPointData().AddArray(self.P_tube_radius)
        self.tubePolyData_P.GetPointData().SetActiveScalars("TubeRadius")
        # Tube filter:
        self.tuber_P = vtk.vtkTubeFilter();
        self.tuber_P.SetInputData(self.tubePolyData_P);
        self.tuber_P.SetNumberOfSides(6);
        self.tuber_P.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        # Line Mapper
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(self.tubePolyData_P)
        lineMapper.SetScalarRange(self.tubePolyData_P.GetScalarRange())
        # Tube Mapper
        tubeMapper = vtk.vtkPolyDataMapper()
        tubeMapper.SetInputConnection(self.tuber_P.GetOutputPort())
        tubeMapper.ScalarVisibilityOff()
        #tubeMapper.SetColorModeToDefault()
        #tubeMapper.SetScalarRange(self.tubePolyData_P.GetScalarRange())
        # Line Actor
        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        # Tube actor
        self.tubeActor_P = vtk.vtkActor()
        self.tubeActor_P.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_P.SetMapper(tubeMapper)

    def membrane_0(self):
        # Vertices
        self.mem_pts_0 = vtk.vtkPoints()
        self.mem_pts_0.SetDataTypeToFloat()
        self.mem_pts_0.InsertPoint(0,(0.0,0.0,0.0))
        self.mem_pts_0.InsertPoint(1,(0.0578,-0.0145,0.0))
        self.mem_pts_0.InsertPoint(2,(0.1301,-0.0940,0.0))
        self.mem_pts_0.InsertPoint(3,(0.1735,-0.1663,0.0))
        self.mem_pts_0.InsertPoint(4,(0.2024,-0.2386,0.0))
        self.mem_pts_0.InsertPoint(5,(0.2241,-0.3108,0.0))
        self.mem_pts_0.InsertPoint(6,(0.2313,-0.3831,0.0))
        self.mem_pts_0.InsertPoint(7,(0.2458,-0.4554,0.0))
        self.mem_pts_0.InsertPoint(8,(0.2458,-0.5277,0.0))
        self.mem_pts_0.InsertPoint(9,(0.2313,-0.5711,0.0))
        self.mem_pts_0.InsertPoint(10,(0.2747,-0.6723,0.0))
        self.mem_pts_0.InsertPoint(11,(0.3036,-0.7446,0.0))
        self.mem_pts_0.InsertPoint(12,(0.3181,-0.8169,0.0))
        self.mem_pts_0.InsertPoint(13,(0.3398,-0.8892,0.0))
        self.mem_pts_0.InsertPoint(14,(0.3542,-0.9615,0.0))
        self.mem_pts_0.InsertPoint(15,(0.3687,-1.0337,0.0))
        self.mem_pts_0.InsertPoint(16,(0.3831,-1.1060,0.0))
        self.mem_pts_0.InsertPoint(17,(0.3904,-1.1783,0.0))
        self.mem_pts_0.InsertPoint(18,(0.3976,-1.2506,0.0))
        self.mem_pts_0.InsertPoint(19,(0.4048,-1.3229,0.0))
        self.mem_pts_0.InsertPoint(20,(0.4121,-1.3952,0.0))
        self.mem_pts_0.InsertPoint(21,(0.4193,-1.4675,0.0))
        self.mem_pts_0.InsertPoint(22,(0.4193,-1.5398,0.0))
        self.mem_pts_0.InsertPoint(23,(0.4265,-1.6121,0.0))
        self.mem_pts_0.InsertPoint(24,(0.4265,-1.6844,0.0))
        self.mem_pts_0.InsertPoint(25,(0.4337,-1.7566,0.0))
        self.mem_pts_0.InsertPoint(26,(0.4265,-1.8289,0.0))
        self.mem_pts_0.InsertPoint(27,(0.4193,-1.9012,0.0))
        self.mem_pts_0.InsertPoint(28,(0.4121,-1.9735,0.0))
        self.mem_pts_0.InsertPoint(29,(0.4048,-2.0458,0.0))
        self.mem_pts_0.InsertPoint(30,(0.3904,-2.1181,0.0))
        self.mem_pts_0.InsertPoint(31,(0.3759,-2.1904,0.0))
        self.mem_pts_0.InsertPoint(32,(0.3542,-2.2627,0.0))
        self.mem_pts_0.InsertPoint(33,(0.3253,-2.3350,0.0))
        self.mem_pts_0.InsertPoint(34,(0.2819,-2.4073,0.0))
        self.mem_pts_0.InsertPoint(35,(0.2241,-2.4795,0.0))
        self.mem_pts_0.InsertPoint(36,(0.1374,-2.5518,0.0))
        self.mem_pts_0.InsertPoint(37,(0.0867,-2.5880,0.0))
        self.mem_pts_0.InsertPoint(38,(0.0506,-2.6097,0.0))
        self.mem_pts_0.InsertPoint(39,(0.0,-2.6241,0.0))
        self.mem_pts_0.InsertPoint(40,(0.0,-2.6097,0.0))
        self.mem_pts_0.InsertPoint(41,(0.0,-2.5880,0.0))
        self.mem_pts_0.InsertPoint(42,(0.0,-2.5518,0.0))
        self.mem_pts_0.InsertPoint(43,(0.0,-2.4795,0.0))
        self.mem_pts_0.InsertPoint(44,(0.0,-2.4073,0.0))
        self.mem_pts_0.InsertPoint(45,(0.0,-2.3350,0.0))
        self.mem_pts_0.InsertPoint(46,(0.0,-2.2627,0.0))
        self.mem_pts_0.InsertPoint(47,(0.0,-2.1904,0.0))
        self.mem_pts_0.InsertPoint(48,(0.0,-2.1181,0.0))
        self.mem_pts_0.InsertPoint(49,(0.0,-2.0458,0.0))
        self.mem_pts_0.InsertPoint(50,(0.0,-1.9735,0.0))
        self.mem_pts_0.InsertPoint(51,(0.0,-1.9012,0.0))
        self.mem_pts_0.InsertPoint(52,(0.0,-1.8289,0.0))
        self.mem_pts_0.InsertPoint(53,(0.0,-1.7566,0.0))
        self.mem_pts_0.InsertPoint(54,(0.0,-1.6844,0.0))
        self.mem_pts_0.InsertPoint(55,(0.0,-1.6121,0.0))
        self.mem_pts_0.InsertPoint(56,(0.0,-1.5398,0.0))
        self.mem_pts_0.InsertPoint(57,(0.0,-1.4675,0.0))
        self.mem_pts_0.InsertPoint(58,(0.0,-1.3952,0.0))
        self.mem_pts_0.InsertPoint(59,(0.0,-1.3229,0.0))
        self.mem_pts_0.InsertPoint(60,(0.0,-1.2506,0.0))
        self.mem_pts_0.InsertPoint(61,(0.0,-1.1783,0.0))
        self.mem_pts_0.InsertPoint(62,(0.0,-1.1060,0.0))
        self.mem_pts_0.InsertPoint(63,(0.0,-1.0337,0.0))
        self.mem_pts_0.InsertPoint(64,(0.0,-0.9615,0.0))
        self.mem_pts_0.InsertPoint(65,(0.0,-0.8892,0.0))
        self.mem_pts_0.InsertPoint(66,(0.0,-0.8169,0.0))
        self.mem_pts_0.InsertPoint(67,(0.0,-0.7446,0.0))
        self.mem_pts_0.InsertPoint(68,(0.0,-0.6723,0.0))
        self.mem_pts_0.InsertPoint(69,(0.0,-0.5711,0.0))
        self.mem_pts_0.InsertPoint(70,(0.0,-0.5277,0.0))
        self.mem_pts_0.InsertPoint(71,(0.0,-0.4554,0.0))
        self.mem_pts_0.InsertPoint(72,(0.0,-0.3831,0.0))
        self.mem_pts_0.InsertPoint(73,(0.0,-0.3108,0.0))
        self.mem_pts_0.InsertPoint(74,(0.0,-0.2386,0.0))
        self.mem_pts_0.InsertPoint(75,(0.0,-0.1663,0.0))
        self.mem_pts_0.InsertPoint(76,(0.0,-0.0940,0.0))
        self.mem_pts_0.InsertPoint(77,(0.0,-0.0145,0.0))
        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,77)
        triangle_2.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,76)
        triangle_4.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,75)
        triangle_6.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,74)
        triangle_8.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,73)
        triangle_10.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,72)
        triangle_12.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,71)
        triangle_14.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,70)
        triangle_16.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,69)
        triangle_18.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,68)
        triangle_20.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,67)
        triangle_22.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,66)
        triangle_24.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,65)
        triangle_26.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,64)
        triangle_28.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,63)
        triangle_30.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,62)
        triangle_32.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,61)
        triangle_34.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,60)
        triangle_36.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,59)
        triangle_38.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,58)
        triangle_40.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,57)
        triangle_42.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_42)
        # triangle 43
        triangle_43 = vtk.vtkTriangle()
        triangle_43.GetPointIds().SetId(0,22)
        triangle_43.GetPointIds().SetId(1,23)
        triangle_43.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_43)
        # triangle 44
        triangle_44 = vtk.vtkTriangle()
        triangle_44.GetPointIds().SetId(0,23)
        triangle_44.GetPointIds().SetId(1,56)
        triangle_44.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_44)
        # triangle 45
        triangle_45 = vtk.vtkTriangle()
        triangle_45.GetPointIds().SetId(0,23)
        triangle_45.GetPointIds().SetId(1,24)
        triangle_45.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_45)
        # triangle 46
        triangle_46 = vtk.vtkTriangle()
        triangle_46.GetPointIds().SetId(0,24)
        triangle_46.GetPointIds().SetId(1,55)
        triangle_46.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_46)
        # triangle 47
        triangle_47 = vtk.vtkTriangle()
        triangle_47.GetPointIds().SetId(0,24)
        triangle_47.GetPointIds().SetId(1,25)
        triangle_47.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_47)
        # triangle 48
        triangle_48 = vtk.vtkTriangle()
        triangle_48.GetPointIds().SetId(0,25)
        triangle_48.GetPointIds().SetId(1,54)
        triangle_48.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_48)
        # triangle 49
        triangle_49 = vtk.vtkTriangle()
        triangle_49.GetPointIds().SetId(0,25)
        triangle_49.GetPointIds().SetId(1,26)
        triangle_49.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_49)
        # triangle 50
        triangle_50 = vtk.vtkTriangle()
        triangle_50.GetPointIds().SetId(0,26)
        triangle_50.GetPointIds().SetId(1,53)
        triangle_50.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_50)
        # triangle 51
        triangle_51 = vtk.vtkTriangle()
        triangle_51.GetPointIds().SetId(0,26)
        triangle_51.GetPointIds().SetId(1,27)
        triangle_51.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_51)
        # triangle 52
        triangle_52 = vtk.vtkTriangle()
        triangle_52.GetPointIds().SetId(0,27)
        triangle_52.GetPointIds().SetId(1,52)
        triangle_52.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_52)
        # triangle 53
        triangle_53 = vtk.vtkTriangle()
        triangle_53.GetPointIds().SetId(0,27)
        triangle_53.GetPointIds().SetId(1,28)
        triangle_53.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_53)
        # triangle 54
        triangle_54 = vtk.vtkTriangle()
        triangle_54.GetPointIds().SetId(0,28)
        triangle_54.GetPointIds().SetId(1,51)
        triangle_54.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_54)
        # triangle 55
        triangle_55 = vtk.vtkTriangle()
        triangle_55.GetPointIds().SetId(0,28)
        triangle_55.GetPointIds().SetId(1,29)
        triangle_55.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_55)
        # triangle 56
        triangle_56 = vtk.vtkTriangle()
        triangle_56.GetPointIds().SetId(0,29)
        triangle_56.GetPointIds().SetId(1,50)
        triangle_56.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_56)
        # triangle 57
        triangle_57 = vtk.vtkTriangle()
        triangle_57.GetPointIds().SetId(0,29)
        triangle_57.GetPointIds().SetId(1,30)
        triangle_57.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_57)
        # triangle 58
        triangle_58 = vtk.vtkTriangle()
        triangle_58.GetPointIds().SetId(0,30)
        triangle_58.GetPointIds().SetId(1,49)
        triangle_58.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_58)
        # triangle 59
        triangle_59 = vtk.vtkTriangle()
        triangle_59.GetPointIds().SetId(0,30)
        triangle_59.GetPointIds().SetId(1,31)
        triangle_59.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_59)
        # triangle 60
        triangle_60 = vtk.vtkTriangle()
        triangle_60.GetPointIds().SetId(0,31)
        triangle_60.GetPointIds().SetId(1,48)
        triangle_60.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_60)
        # triangle 61
        triangle_61 = vtk.vtkTriangle()
        triangle_61.GetPointIds().SetId(0,31)
        triangle_61.GetPointIds().SetId(1,32)
        triangle_61.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_61)
        # triangle 62
        triangle_62 = vtk.vtkTriangle()
        triangle_62.GetPointIds().SetId(0,32)
        triangle_62.GetPointIds().SetId(1,47)
        triangle_62.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_62)
        # triangle 63
        triangle_63 = vtk.vtkTriangle()
        triangle_63.GetPointIds().SetId(0,32)
        triangle_63.GetPointIds().SetId(1,33)
        triangle_63.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_63)
        # triangle 64
        triangle_64 = vtk.vtkTriangle()
        triangle_64.GetPointIds().SetId(0,33)
        triangle_64.GetPointIds().SetId(1,46)
        triangle_64.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_64)
        # triangle 65
        triangle_65 = vtk.vtkTriangle()
        triangle_65.GetPointIds().SetId(0,33)
        triangle_65.GetPointIds().SetId(1,34)
        triangle_65.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_65)
        # triangle 66
        triangle_66 = vtk.vtkTriangle()
        triangle_66.GetPointIds().SetId(0,34)
        triangle_66.GetPointIds().SetId(1,45)
        triangle_66.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_66)
        # triangle 67
        triangle_67 = vtk.vtkTriangle()
        triangle_67.GetPointIds().SetId(0,34)
        triangle_67.GetPointIds().SetId(1,35)
        triangle_67.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_67)
        # triangle 68
        triangle_68 = vtk.vtkTriangle()
        triangle_68.GetPointIds().SetId(0,35)
        triangle_68.GetPointIds().SetId(1,44)
        triangle_68.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_68)
        # triangle 69
        triangle_69 = vtk.vtkTriangle()
        triangle_69.GetPointIds().SetId(0,35)
        triangle_69.GetPointIds().SetId(1,36)
        triangle_69.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_69)
        # triangle 70
        triangle_70 = vtk.vtkTriangle()
        triangle_70.GetPointIds().SetId(0,36)
        triangle_70.GetPointIds().SetId(1,43)
        triangle_70.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_70)
        # triangle 71
        triangle_71 = vtk.vtkTriangle()
        triangle_71.GetPointIds().SetId(0,36)
        triangle_71.GetPointIds().SetId(1,37)
        triangle_71.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_71)
        # triangle 72
        triangle_72 = vtk.vtkTriangle()
        triangle_72.GetPointIds().SetId(0,37)
        triangle_72.GetPointIds().SetId(1,42)
        triangle_72.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_72)
        # triangle 73
        triangle_73 = vtk.vtkTriangle()
        triangle_73.GetPointIds().SetId(0,37)
        triangle_73.GetPointIds().SetId(1,38)
        triangle_73.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_73)
        # triangle 74
        triangle_74 = vtk.vtkTriangle()
        triangle_74.GetPointIds().SetId(0,38)
        triangle_74.GetPointIds().SetId(1,41)
        triangle_74.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_74)
        # triangle 75
        triangle_75 = vtk.vtkTriangle()
        triangle_75.GetPointIds().SetId(0,38)
        triangle_75.GetPointIds().SetId(1,39)
        triangle_75.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_75)

        self.mem_0 = vtk.vtkPolyData()
        self.mem_0.SetPoints(self.mem_pts_0)
        self.mem_0.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_0)

        self.membrane_Actor_0 = vtk.vtkActor()
        self.membrane_Actor_0.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_0.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_0.ForceTranslucentOn()
        self.membrane_Actor_0.SetMapper(Mapper)

    def membrane_1(self):
        # Vertices
        self.mem_pts_1 = vtk.vtkPoints()
        self.mem_pts_1.SetDataTypeToFloat()
        self.mem_pts_1.InsertPoint(0,(0.0,0.0,0.0))
        self.mem_pts_1.InsertPoint(1,(0.0,-0.0145,0.0))
        self.mem_pts_1.InsertPoint(2,(0.0,-0.0940,0.0))
        self.mem_pts_1.InsertPoint(3,(0.0,-0.1663,0.0))
        self.mem_pts_1.InsertPoint(4,(0.0,-0.2386,0.0))
        self.mem_pts_1.InsertPoint(5,(0.0,-0.3108,0.0))
        self.mem_pts_1.InsertPoint(6,(0.0,-0.3831,0.0))
        self.mem_pts_1.InsertPoint(7,(0.0,-0.4554,0.0))
        self.mem_pts_1.InsertPoint(8,(0.0,-0.5277,0.0))
        self.mem_pts_1.InsertPoint(9,(0.0,-0.5711,0.0))
        self.mem_pts_1.InsertPoint(10,(0.0,-0.6723,0.0))
        self.mem_pts_1.InsertPoint(11,(0.0,-0.7446,0.0))
        self.mem_pts_1.InsertPoint(12,(0.0,-0.8169,0.0))
        self.mem_pts_1.InsertPoint(13,(0.0,-0.8892,0.0))
        self.mem_pts_1.InsertPoint(14,(0.0,-0.9615,0.0))
        self.mem_pts_1.InsertPoint(15,(0.0,-1.0337,0.0))
        self.mem_pts_1.InsertPoint(16,(0.0,-1.1060,0.0))
        self.mem_pts_1.InsertPoint(17,(0.0,-1.1783,0.0))
        self.mem_pts_1.InsertPoint(18,(0.0,-1.2506,0.0))
        self.mem_pts_1.InsertPoint(19,(0.0,-1.3229,0.0))
        self.mem_pts_1.InsertPoint(20,(0.0,-1.3952,0.0))
        self.mem_pts_1.InsertPoint(21,(0.0,-1.4675,0.0))
        self.mem_pts_1.InsertPoint(22,(0.0,-1.5398,0.0))
        self.mem_pts_1.InsertPoint(23,(0.0,-1.6121,0.0))
        self.mem_pts_1.InsertPoint(24,(0.0,-1.6844,0.0))
        self.mem_pts_1.InsertPoint(25,(0.0,-1.7566,0.0))
        self.mem_pts_1.InsertPoint(26,(0.0,-1.8289,0.0))
        self.mem_pts_1.InsertPoint(27,(0.0,-1.9012,0.0))
        self.mem_pts_1.InsertPoint(28,(0.0,-1.9735,0.0))
        self.mem_pts_1.InsertPoint(29,(0.0,-2.0458,0.0))
        self.mem_pts_1.InsertPoint(30,(0.0,-2.1181,0.0))
        self.mem_pts_1.InsertPoint(31,(0.0,-2.1904,0.0))
        self.mem_pts_1.InsertPoint(32,(0.0,-2.2627,0.0))
        self.mem_pts_1.InsertPoint(33,(0.0,-2.3350,0.0))
        self.mem_pts_1.InsertPoint(34,(0.0,-2.4073,0.0))
        self.mem_pts_1.InsertPoint(35,(0.0,-2.4795,0.0))
        self.mem_pts_1.InsertPoint(36,(0.0,-2.5518,0.0))
        self.mem_pts_1.InsertPoint(37,(0.0,-2.5880,0.0))
        self.mem_pts_1.InsertPoint(38,(0.0,-2.6097,0.0))
        self.mem_pts_1.InsertPoint(39,(0.0,-2.6241,0.0)) # tip
        self.mem_pts_1.InsertPoint(40,(-0.1301,-2.6097,0.0))
        self.mem_pts_1.InsertPoint(41,(-0.1880,-2.5880,0.0))
        self.mem_pts_1.InsertPoint(42,(-0.2313,-2.5518,0.0))
        self.mem_pts_1.InsertPoint(43,(-0.2169,-2.4795,0.0))
        self.mem_pts_1.InsertPoint(44,(-0.2169,-2.4073,0.0))
        self.mem_pts_1.InsertPoint(45,(-0.2096,-2.3350,0.0))
        self.mem_pts_1.InsertPoint(46,(-0.2096,-2.2627,0.0))
        self.mem_pts_1.InsertPoint(47,(-0.2024,-2.1904,0.0))
        self.mem_pts_1.InsertPoint(48,(-0.2024,-2.1181,0.0))
        self.mem_pts_1.InsertPoint(49,(-0.1952,-2.0458,0.0))
        self.mem_pts_1.InsertPoint(50,(-0.1952,-1.9735,0.0))
        self.mem_pts_1.InsertPoint(51,(-0.1876,-1.9012,0.0))
        self.mem_pts_1.InsertPoint(52,(-0.1876,-1.8289,0.0))
        self.mem_pts_1.InsertPoint(53,(-0.1807,-1.7566,0.0))
        self.mem_pts_1.InsertPoint(54,(-0.1807,-1.6844,0.0))
        self.mem_pts_1.InsertPoint(55,(-0.1735,-1.6121,0.0))
        self.mem_pts_1.InsertPoint(56,(-0.1735,-1.5398,0.0))
        self.mem_pts_1.InsertPoint(57,(-0.1663,-1.4675,0.0))
        self.mem_pts_1.InsertPoint(58,(-0.1663,-1.3952,0.0))
        self.mem_pts_1.InsertPoint(59,(-0.1590,-1.3229,0.0))
        self.mem_pts_1.InsertPoint(60,(-0.1590,-1.2506,0.0))
        self.mem_pts_1.InsertPoint(61,(-0.1518,-1.1783,0.0))
        self.mem_pts_1.InsertPoint(62,(-0.1518,-1.1060,0.0))
        self.mem_pts_1.InsertPoint(63,(-0.1446,-1.0337,0.0))
        self.mem_pts_1.InsertPoint(64,(-0.1446,-0.9615,0.0))
        self.mem_pts_1.InsertPoint(65,(-0.1374,-0.8892,0.0))
        self.mem_pts_1.InsertPoint(66,(-0.1374,-0.8169,0.0))
        self.mem_pts_1.InsertPoint(67,(-0.1301,-0.7446,0.0))
        self.mem_pts_1.InsertPoint(68,(-0.1301,-0.6723,0.0))
        self.mem_pts_1.InsertPoint(69,(-0.1229,-0.5711,0.0))
        self.mem_pts_1.InsertPoint(70,(-0.1229,-0.5277,0.0))
        self.mem_pts_1.InsertPoint(71,(-0.1157,-0.4554,0.0))
        self.mem_pts_1.InsertPoint(72,(-0.1157,-0.3831,0.0))
        self.mem_pts_1.InsertPoint(73,(-0.1084,-0.3108,0.0))
        self.mem_pts_1.InsertPoint(74,(-0.1084,-0.2386,0.0))
        self.mem_pts_1.InsertPoint(75,(-0.1012,-0.1663,0.0))
        self.mem_pts_1.InsertPoint(76,(-0.1012,-0.0940,0.0))
        self.mem_pts_1.InsertPoint(77,(-0.0940,-0.0145,0.0))
        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,77)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,77)
        triangle_2.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,76)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,76)
        triangle_4.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,75)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,75)
        triangle_6.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,74)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,74)
        triangle_8.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,73)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,73)
        triangle_10.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,72)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,72)
        triangle_12.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,71)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,71)
        triangle_14.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,70)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,70)
        triangle_16.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,69)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,69)
        triangle_18.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,68)
        triangle_20.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,67)
        triangle_22.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,66)
        triangle_24.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,65)
        triangle_26.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,64)
        triangle_28.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,63)
        triangle_30.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,62)
        triangle_32.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,61)
        triangle_34.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,60)
        triangle_36.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,59)
        triangle_38.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,58)
        triangle_40.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,57)
        triangle_42.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_42)
        # triangle 43
        triangle_43 = vtk.vtkTriangle()
        triangle_43.GetPointIds().SetId(0,22)
        triangle_43.GetPointIds().SetId(1,23)
        triangle_43.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_43)
        # triangle 44
        triangle_44 = vtk.vtkTriangle()
        triangle_44.GetPointIds().SetId(0,23)
        triangle_44.GetPointIds().SetId(1,56)
        triangle_44.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_44)
        # triangle 45
        triangle_45 = vtk.vtkTriangle()
        triangle_45.GetPointIds().SetId(0,23)
        triangle_45.GetPointIds().SetId(1,24)
        triangle_45.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_45)
        # triangle 46
        triangle_46 = vtk.vtkTriangle()
        triangle_46.GetPointIds().SetId(0,24)
        triangle_46.GetPointIds().SetId(1,55)
        triangle_46.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_46)
        # triangle 47
        triangle_47 = vtk.vtkTriangle()
        triangle_47.GetPointIds().SetId(0,24)
        triangle_47.GetPointIds().SetId(1,25)
        triangle_47.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_47)
        # triangle 48
        triangle_48 = vtk.vtkTriangle()
        triangle_48.GetPointIds().SetId(0,25)
        triangle_48.GetPointIds().SetId(1,54)
        triangle_48.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_48)
        # triangle 49
        triangle_49 = vtk.vtkTriangle()
        triangle_49.GetPointIds().SetId(0,25)
        triangle_49.GetPointIds().SetId(1,26)
        triangle_49.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_49)
        # triangle 50
        triangle_50 = vtk.vtkTriangle()
        triangle_50.GetPointIds().SetId(0,26)
        triangle_50.GetPointIds().SetId(1,53)
        triangle_50.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_50)
        # triangle 51
        triangle_51 = vtk.vtkTriangle()
        triangle_51.GetPointIds().SetId(0,26)
        triangle_51.GetPointIds().SetId(1,27)
        triangle_51.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_51)
        # triangle 52
        triangle_52 = vtk.vtkTriangle()
        triangle_52.GetPointIds().SetId(0,27)
        triangle_52.GetPointIds().SetId(1,52)
        triangle_52.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_52)
        # triangle 53
        triangle_53 = vtk.vtkTriangle()
        triangle_53.GetPointIds().SetId(0,27)
        triangle_53.GetPointIds().SetId(1,28)
        triangle_53.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_53)
        # triangle 54
        triangle_54 = vtk.vtkTriangle()
        triangle_54.GetPointIds().SetId(0,28)
        triangle_54.GetPointIds().SetId(1,51)
        triangle_54.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_54)
        # triangle 55
        triangle_55 = vtk.vtkTriangle()
        triangle_55.GetPointIds().SetId(0,28)
        triangle_55.GetPointIds().SetId(1,29)
        triangle_55.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_55)
        # triangle 56
        triangle_56 = vtk.vtkTriangle()
        triangle_56.GetPointIds().SetId(0,29)
        triangle_56.GetPointIds().SetId(1,50)
        triangle_56.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_56)
        # triangle 57
        triangle_57 = vtk.vtkTriangle()
        triangle_57.GetPointIds().SetId(0,29)
        triangle_57.GetPointIds().SetId(1,30)
        triangle_57.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_57)
        # triangle 58
        triangle_58 = vtk.vtkTriangle()
        triangle_58.GetPointIds().SetId(0,30)
        triangle_58.GetPointIds().SetId(1,49)
        triangle_58.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_58)
        # triangle 59
        triangle_59 = vtk.vtkTriangle()
        triangle_59.GetPointIds().SetId(0,30)
        triangle_59.GetPointIds().SetId(1,31)
        triangle_59.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_59)
        # triangle 60
        triangle_60 = vtk.vtkTriangle()
        triangle_60.GetPointIds().SetId(0,31)
        triangle_60.GetPointIds().SetId(1,48)
        triangle_60.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_60)
        # triangle 61
        triangle_61 = vtk.vtkTriangle()
        triangle_61.GetPointIds().SetId(0,31)
        triangle_61.GetPointIds().SetId(1,32)
        triangle_61.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_61)
        # triangle 62
        triangle_62 = vtk.vtkTriangle()
        triangle_62.GetPointIds().SetId(0,32)
        triangle_62.GetPointIds().SetId(1,47)
        triangle_62.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_62)
        # triangle 63
        triangle_63 = vtk.vtkTriangle()
        triangle_63.GetPointIds().SetId(0,32)
        triangle_63.GetPointIds().SetId(1,33)
        triangle_63.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_63)
        # triangle 64
        triangle_64 = vtk.vtkTriangle()
        triangle_64.GetPointIds().SetId(0,33)
        triangle_64.GetPointIds().SetId(1,46)
        triangle_64.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_64)
        # triangle 65
        triangle_65 = vtk.vtkTriangle()
        triangle_65.GetPointIds().SetId(0,33)
        triangle_65.GetPointIds().SetId(1,34)
        triangle_65.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_65)
        # triangle 66
        triangle_66 = vtk.vtkTriangle()
        triangle_66.GetPointIds().SetId(0,34)
        triangle_66.GetPointIds().SetId(1,45)
        triangle_66.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_66)
        # triangle 67
        triangle_67 = vtk.vtkTriangle()
        triangle_67.GetPointIds().SetId(0,34)
        triangle_67.GetPointIds().SetId(1,35)
        triangle_67.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_67)
        # triangle 68
        triangle_68 = vtk.vtkTriangle()
        triangle_68.GetPointIds().SetId(0,35)
        triangle_68.GetPointIds().SetId(1,44)
        triangle_68.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_68)
        # triangle 69
        triangle_69 = vtk.vtkTriangle()
        triangle_69.GetPointIds().SetId(0,35)
        triangle_69.GetPointIds().SetId(1,36)
        triangle_69.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_69)
        # triangle 70
        triangle_70 = vtk.vtkTriangle()
        triangle_70.GetPointIds().SetId(0,36)
        triangle_70.GetPointIds().SetId(1,43)
        triangle_70.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_70)
        # triangle 71
        triangle_71 = vtk.vtkTriangle()
        triangle_71.GetPointIds().SetId(0,36)
        triangle_71.GetPointIds().SetId(1,37)
        triangle_71.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_71)
        # triangle 72
        triangle_72 = vtk.vtkTriangle()
        triangle_72.GetPointIds().SetId(0,37)
        triangle_72.GetPointIds().SetId(1,42)
        triangle_72.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_72)
        # triangle 73
        triangle_73 = vtk.vtkTriangle()
        triangle_73.GetPointIds().SetId(0,37)
        triangle_73.GetPointIds().SetId(1,38)
        triangle_73.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_73)
        # triangle 74
        triangle_74 = vtk.vtkTriangle()
        triangle_74.GetPointIds().SetId(0,38)
        triangle_74.GetPointIds().SetId(1,41)
        triangle_74.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_74)
        # triangle 75
        triangle_75 = vtk.vtkTriangle()
        triangle_75.GetPointIds().SetId(0,38)
        triangle_75.GetPointIds().SetId(1,39)
        triangle_75.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_75)

        self.mem_1 = vtk.vtkPolyData()
        self.mem_1.SetPoints(self.mem_pts_1)
        self.mem_1.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_1)

        self.membrane_Actor_1 = vtk.vtkActor()
        self.membrane_Actor_1.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_1.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_1.ForceTranslucentOn()
        self.membrane_Actor_1.SetMapper(Mapper)

    def membrane_2(self):
        # Vertices
        self.mem_pts_2 = vtk.vtkPoints()
        self.mem_pts_2.SetDataTypeToFloat()
        self.mem_pts_2.InsertPoint(0,(-0.0867,-0.0145,0.0))
        self.mem_pts_2.InsertPoint(1,(-0.1157,-0.0940,0.0))
        self.mem_pts_2.InsertPoint(2,(-0.1518,-0.1663,0.0))
        self.mem_pts_2.InsertPoint(3,(-0.1735,-0.2386,0.0))
        self.mem_pts_2.InsertPoint(4,(-0.2024,-0.3108,0.0))
        self.mem_pts_2.InsertPoint(5,(-0.2313,-0.3831,0.0))
        self.mem_pts_2.InsertPoint(6,(-0.2530,-0.4554,0.0))
        self.mem_pts_2.InsertPoint(7,(-0.2819,-0.5277,0.0))
        self.mem_pts_2.InsertPoint(8,(-0.3109,-0.5711,0.0))
        self.mem_pts_2.InsertPoint(9,(-0.3398,-0.6723,0.0))
        self.mem_pts_2.InsertPoint(10,(-0.3687,-0.7446,0.0))
        self.mem_pts_2.InsertPoint(11,(-0.3976,-0.8169,0.0))
        self.mem_pts_2.InsertPoint(12,(-0.4193,-0.8892,0.0))
        self.mem_pts_2.InsertPoint(13,(-0.4554,-0.9615,0.0))
        self.mem_pts_2.InsertPoint(14,(-0.4771,-1.0337,0.0))
        self.mem_pts_2.InsertPoint(15,(-0.5060,-1.1060,0.0))
        self.mem_pts_2.InsertPoint(16,(-0.5350,-1.1783,0.0))
        self.mem_pts_2.InsertPoint(17,(-0.5566,-1.2506,0.0))
        self.mem_pts_2.InsertPoint(18,(-0.5856,-1.3229,0.0))
        self.mem_pts_2.InsertPoint(19,(-0.6145,-1.3952,0.0))
        self.mem_pts_2.InsertPoint(20,(-0.6434,-1.4675,0.0))
        self.mem_pts_2.InsertPoint(21,(-0.6723,-1.5398,0.0))
        self.mem_pts_2.InsertPoint(22,(-0.6940,-1.6121,0.0))
        self.mem_pts_2.InsertPoint(23,(-0.6940,-1.6844,0.0))
        self.mem_pts_2.InsertPoint(24,(-0.6868,-1.7566,0.0))
        self.mem_pts_2.InsertPoint(25,(-0.6578,-1.8289,0.0))
        self.mem_pts_2.InsertPoint(26,(-0.6362,-1.9012,0.0))
        self.mem_pts_2.InsertPoint(27,(-0.6145,-1.9735,0.0))
        self.mem_pts_2.InsertPoint(28,(-0.5856,-2.0458,0.0))
        self.mem_pts_2.InsertPoint(29,(-0.5566,-2.1181,0.0))
        self.mem_pts_2.InsertPoint(30,(-0.5205,-2.1904,0.0))
        self.mem_pts_2.InsertPoint(31,(-0.4843,-2.2627,0.0))
        self.mem_pts_2.InsertPoint(32,(-0.4337,-2.3350,0.0))
        self.mem_pts_2.InsertPoint(33,(-0.3831,-2.4073,0.0))
        self.mem_pts_2.InsertPoint(34,(-0.2313,-2.5518,0.0))
        self.mem_pts_2.InsertPoint(35,(-0.2169,-2.4795,0.0))
        self.mem_pts_2.InsertPoint(36,(-0.2169,-2.4073,0.0))
        self.mem_pts_2.InsertPoint(37,(-0.2096,-2.3350,0.0))
        self.mem_pts_2.InsertPoint(38,(-0.2096,-2.2627,0.0))
        self.mem_pts_2.InsertPoint(39,(-0.2024,-2.1904,0.0))
        self.mem_pts_2.InsertPoint(40,(-0.2024,-2.1181,0.0))
        self.mem_pts_2.InsertPoint(41,(-0.1952,-2.0458,0.0))
        self.mem_pts_2.InsertPoint(42,(-0.1952,-1.9735,0.0))
        self.mem_pts_2.InsertPoint(43,(-0.1876,-1.9012,0.0))
        self.mem_pts_2.InsertPoint(44,(-0.1876,-1.8289,0.0))
        self.mem_pts_2.InsertPoint(45,(-0.1807,-1.7566,0.0))
        self.mem_pts_2.InsertPoint(46,(-0.1807,-1.6844,0.0))
        self.mem_pts_2.InsertPoint(47,(-0.1735,-1.6121,0.0))
        self.mem_pts_2.InsertPoint(48,(-0.1735,-1.5398,0.0))
        self.mem_pts_2.InsertPoint(49,(-0.1663,-1.4675,0.0))
        self.mem_pts_2.InsertPoint(50,(-0.1663,-1.3952,0.0))
        self.mem_pts_2.InsertPoint(51,(-0.1590,-1.3229,0.0))
        self.mem_pts_2.InsertPoint(52,(-0.1590,-1.2506,0.0))
        self.mem_pts_2.InsertPoint(53,(-0.1518,-1.1783,0.0))
        self.mem_pts_2.InsertPoint(54,(-0.1518,-1.1060,0.0))
        self.mem_pts_2.InsertPoint(55,(-0.1446,-1.0337,0.0))
        self.mem_pts_2.InsertPoint(56,(-0.1446,-0.9615,0.0))
        self.mem_pts_2.InsertPoint(57,(-0.1374,-0.8892,0.0))
        self.mem_pts_2.InsertPoint(58,(-0.1374,-0.8169,0.0))
        self.mem_pts_2.InsertPoint(59,(-0.1301,-0.7446,0.0))
        self.mem_pts_2.InsertPoint(60,(-0.1301,-0.6723,0.0))
        self.mem_pts_2.InsertPoint(61,(-0.1229,-0.5711,0.0))
        self.mem_pts_2.InsertPoint(62,(-0.1229,-0.5277,0.0))
        self.mem_pts_2.InsertPoint(63,(-0.1157,-0.4554,0.0))
        self.mem_pts_2.InsertPoint(64,(-0.1157,-0.3831,0.0))
        self.mem_pts_2.InsertPoint(65,(-0.1084,-0.3108,0.0))
        self.mem_pts_2.InsertPoint(66,(-0.1084,-0.2386,0.0))
        self.mem_pts_2.InsertPoint(67,(-0.1012,-0.1663,0.0))
        self.mem_pts_2.InsertPoint(68,(-0.1012,-0.0940,0.0))

        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,68)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,68)
        triangle_2.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,67)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,67)
        triangle_4.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,66)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,66)
        triangle_6.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,65)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,65)
        triangle_8.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,64)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,64)
        triangle_10.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,63)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,63)
        triangle_12.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,62)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,62)
        triangle_14.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,61)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,61)
        triangle_16.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,60)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,60)
        triangle_18.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,59)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,59)
        triangle_20.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,58)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,58)
        triangle_22.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,57)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,57)
        triangle_24.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,56)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,56)
        triangle_26.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,55)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,55)
        triangle_28.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,54)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,54)
        triangle_30.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,53)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,53)
        triangle_32.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,52)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,52)
        triangle_34.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,51)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,51)
        triangle_36.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,50)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,50)
        triangle_38.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,49)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,49)
        triangle_40.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,48)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,48)
        triangle_42.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_42)
        # triangle 43
        triangle_43 = vtk.vtkTriangle()
        triangle_43.GetPointIds().SetId(0,22)
        triangle_43.GetPointIds().SetId(1,23)
        triangle_43.GetPointIds().SetId(2,47)
        triangles.InsertNextCell(triangle_43)
        # triangle 44
        triangle_44 = vtk.vtkTriangle()
        triangle_44.GetPointIds().SetId(0,23)
        triangle_44.GetPointIds().SetId(1,47)
        triangle_44.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_44)
        # triangle 45
        triangle_45 = vtk.vtkTriangle()
        triangle_45.GetPointIds().SetId(0,23)
        triangle_45.GetPointIds().SetId(1,24)
        triangle_45.GetPointIds().SetId(2,46)
        triangles.InsertNextCell(triangle_45)
        # triangle 46
        triangle_46 = vtk.vtkTriangle()
        triangle_46.GetPointIds().SetId(0,24)
        triangle_46.GetPointIds().SetId(1,46)
        triangle_46.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_46)
        # triangle 47
        triangle_47 = vtk.vtkTriangle()
        triangle_47.GetPointIds().SetId(0,24)
        triangle_47.GetPointIds().SetId(1,25)
        triangle_47.GetPointIds().SetId(2,45)
        triangles.InsertNextCell(triangle_47)
        # triangle 48
        triangle_48 = vtk.vtkTriangle()
        triangle_48.GetPointIds().SetId(0,25)
        triangle_48.GetPointIds().SetId(1,45)
        triangle_48.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_48)
        # triangle 49
        triangle_49 = vtk.vtkTriangle()
        triangle_49.GetPointIds().SetId(0,25)
        triangle_49.GetPointIds().SetId(1,26)
        triangle_49.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_49)
        # triangle 50
        triangle_50 = vtk.vtkTriangle()
        triangle_50.GetPointIds().SetId(0,26)
        triangle_50.GetPointIds().SetId(1,44)
        triangle_50.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_50)
        # triangle 51
        triangle_51 = vtk.vtkTriangle()
        triangle_51.GetPointIds().SetId(0,26)
        triangle_51.GetPointIds().SetId(1,27)
        triangle_51.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_51)
        # triangle 52
        triangle_52 = vtk.vtkTriangle()
        triangle_52.GetPointIds().SetId(0,27)
        triangle_52.GetPointIds().SetId(1,43)
        triangle_52.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_52)
        # triangle 53
        triangle_53 = vtk.vtkTriangle()
        triangle_53.GetPointIds().SetId(0,27)
        triangle_53.GetPointIds().SetId(1,28)
        triangle_53.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_53)
        # triangle 54
        triangle_54 = vtk.vtkTriangle()
        triangle_54.GetPointIds().SetId(0,28)
        triangle_54.GetPointIds().SetId(1,42)
        triangle_54.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_54)
        # triangle 55
        triangle_55 = vtk.vtkTriangle()
        triangle_55.GetPointIds().SetId(0,28)
        triangle_55.GetPointIds().SetId(1,29)
        triangle_55.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_55)
        # triangle 56
        triangle_56 = vtk.vtkTriangle()
        triangle_56.GetPointIds().SetId(0,29)
        triangle_56.GetPointIds().SetId(1,41)
        triangle_56.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_56)
        # triangle 57
        triangle_57 = vtk.vtkTriangle()
        triangle_57.GetPointIds().SetId(0,29)
        triangle_57.GetPointIds().SetId(1,30)
        triangle_57.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_57)
        # triangle 58
        triangle_58 = vtk.vtkTriangle()
        triangle_58.GetPointIds().SetId(0,30)
        triangle_58.GetPointIds().SetId(1,40)
        triangle_58.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_58)
        # triangle 59
        triangle_59 = vtk.vtkTriangle()
        triangle_59.GetPointIds().SetId(0,30)
        triangle_59.GetPointIds().SetId(1,31)
        triangle_59.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_59)
        # triangle 60
        triangle_60 = vtk.vtkTriangle()
        triangle_60.GetPointIds().SetId(0,31)
        triangle_60.GetPointIds().SetId(1,39)
        triangle_60.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_60)
        # triangle 61
        triangle_61 = vtk.vtkTriangle()
        triangle_61.GetPointIds().SetId(0,31)
        triangle_61.GetPointIds().SetId(1,32)
        triangle_61.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_61)
        # triangle 62
        triangle_62 = vtk.vtkTriangle()
        triangle_62.GetPointIds().SetId(0,32)
        triangle_62.GetPointIds().SetId(1,38)
        triangle_62.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_62)
        # triangle 63
        triangle_63 = vtk.vtkTriangle()
        triangle_63.GetPointIds().SetId(0,32)
        triangle_63.GetPointIds().SetId(1,33)
        triangle_63.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_63)
        # triangle 64
        triangle_64 = vtk.vtkTriangle()
        triangle_64.GetPointIds().SetId(0,33)
        triangle_64.GetPointIds().SetId(1,37)
        triangle_64.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_64)
        # triangle 65
        triangle_65 = vtk.vtkTriangle()
        triangle_65.GetPointIds().SetId(0,33)
        triangle_65.GetPointIds().SetId(1,34)
        triangle_65.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_65)
        # triangle 66
        triangle_66 = vtk.vtkTriangle()
        triangle_66.GetPointIds().SetId(0,34)
        triangle_66.GetPointIds().SetId(1,36)
        triangle_66.GetPointIds().SetId(2,35)
        triangles.InsertNextCell(triangle_66)
        # triangle 67
        triangle_67 = vtk.vtkTriangle()
        triangle_67.GetPointIds().SetId(0,34)
        triangle_67.GetPointIds().SetId(1,35)
        triangle_67.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_67)        

        self.mem_2 = vtk.vtkPolyData()
        self.mem_2.SetPoints(self.mem_pts_2)
        self.mem_2.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_2)

        self.membrane_Actor_2 = vtk.vtkActor()
        self.membrane_Actor_2.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_2.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_2.ForceTranslucentOn()
        self.membrane_Actor_2.SetMapper(Mapper)

    def membrane_3(self):
        # Vertices
        self.mem_pts_3 = vtk.vtkPoints()
        self.mem_pts_3.SetDataTypeToFloat()
        self.mem_pts_3.InsertPoint(0,(-0.0867,-0.0145,0.0))
        self.mem_pts_3.InsertPoint(1,(-0.1157,-0.0940,0.0))
        self.mem_pts_3.InsertPoint(2,(-0.1518,-0.1663,0.0))
        self.mem_pts_3.InsertPoint(3,(-0.1735,-0.2386,0.0))
        self.mem_pts_3.InsertPoint(4,(-0.2024,-0.3108,0.0))
        self.mem_pts_3.InsertPoint(5,(-0.2313,-0.3831,0.0))
        self.mem_pts_3.InsertPoint(6,(-0.2530,-0.4554,0.0))
        self.mem_pts_3.InsertPoint(7,(-0.2819,-0.5277,0.0))
        self.mem_pts_3.InsertPoint(8,(-0.3109,-0.5711,0.0))
        self.mem_pts_3.InsertPoint(9,(-0.3398,-0.6723,0.0))
        self.mem_pts_3.InsertPoint(10,(-0.3687,-0.7446,0.0))
        self.mem_pts_3.InsertPoint(11,(-0.3976,-0.8169,0.0))
        self.mem_pts_3.InsertPoint(12,(-0.4193,-0.8892,0.0))
        self.mem_pts_3.InsertPoint(13,(-0.4554,-0.9615,0.0))
        self.mem_pts_3.InsertPoint(14,(-0.4771,-1.0337,0.0))
        self.mem_pts_3.InsertPoint(15,(-0.5060,-1.1060,0.0))
        self.mem_pts_3.InsertPoint(16,(-0.5350,-1.1783,0.0))
        self.mem_pts_3.InsertPoint(17,(-0.5566,-1.2506,0.0))
        self.mem_pts_3.InsertPoint(18,(-0.5856,-1.3229,0.0))
        self.mem_pts_3.InsertPoint(19,(-0.6145,-1.3952,0.0))
        self.mem_pts_3.InsertPoint(20,(-0.6434,-1.4675,0.0))
        self.mem_pts_3.InsertPoint(21,(-0.6723,-1.5398,0.0))
        self.mem_pts_3.InsertPoint(22,(-0.6940,-1.6121,0.0))
        self.mem_pts_3.InsertPoint(23,(-0.7229,-1.5398,0.0))
        self.mem_pts_3.InsertPoint(24,(-0.7446,-1.4675,0.0))
        self.mem_pts_3.InsertPoint(25,(-0.7518,-1.3952,0.0))
        self.mem_pts_3.InsertPoint(26,(-0.7663,-1.3229,0.0))
        self.mem_pts_3.InsertPoint(27,(-0.7735,-1.2506,0.0))
        self.mem_pts_3.InsertPoint(28,(-0.7807,-1.1783,0.0))
        self.mem_pts_3.InsertPoint(29,(-0.7807,-1.1060,0.0))
        self.mem_pts_3.InsertPoint(30,(-0.7807,-1.0337,0.0))
        self.mem_pts_3.InsertPoint(31,(-0.7880,-0.9615,0.0))
        self.mem_pts_3.InsertPoint(32,(-0.7880,-0.8892,0.0))
        self.mem_pts_3.InsertPoint(33,(-0.7880,-0.8169,0.0))
        self.mem_pts_3.InsertPoint(34,(-0.7807,-0.7446,0.0))
        self.mem_pts_3.InsertPoint(35,(-0.7663,-0.6723,0.0))
        self.mem_pts_3.InsertPoint(36,(-0.7590,-0.6000,0.0))
        self.mem_pts_3.InsertPoint(37,(-0.7446,-0.5277,0.0))
        self.mem_pts_3.InsertPoint(38,(-0.7229,-0.4554,0.0))
        self.mem_pts_3.InsertPoint(39,(-0.6940,-0.3831,0.0))
        self.mem_pts_3.InsertPoint(40,(-0.6434,-0.3108,0.0))
        self.mem_pts_3.InsertPoint(41,(-0.5566,-0.2386,0.0))
        self.mem_pts_3.InsertPoint(42,(-0.3831,-0.1663,0.0))
        self.mem_pts_3.InsertPoint(43,(-0.2169,-0.0940,0.0))
        self.mem_pts_3.InsertPoint(44,(-0.1157,-0.0217,0.0))

        # Cell array
        triangles = vtk.vtkCellArray()
        # triangle 0
        triangle_0 = vtk.vtkTriangle()
        triangle_0.GetPointIds().SetId(0,0)
        triangle_0.GetPointIds().SetId(1,1)
        triangle_0.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_0)
        # triangle 1
        triangle_1 = vtk.vtkTriangle()
        triangle_1.GetPointIds().SetId(0,1)
        triangle_1.GetPointIds().SetId(1,2)
        triangle_1.GetPointIds().SetId(2,44)
        triangles.InsertNextCell(triangle_1)
        # triangle 2
        triangle_2 = vtk.vtkTriangle()
        triangle_2.GetPointIds().SetId(0,2)
        triangle_2.GetPointIds().SetId(1,44)
        triangle_2.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_2)
        # triangle 3
        triangle_3 = vtk.vtkTriangle()
        triangle_3.GetPointIds().SetId(0,2)
        triangle_3.GetPointIds().SetId(1,3)
        triangle_3.GetPointIds().SetId(2,43)
        triangles.InsertNextCell(triangle_3)
        # triangle 4
        triangle_4 = vtk.vtkTriangle()
        triangle_4.GetPointIds().SetId(0,3)
        triangle_4.GetPointIds().SetId(1,43)
        triangle_4.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_4)
        # triangle 5
        triangle_5 = vtk.vtkTriangle()
        triangle_5.GetPointIds().SetId(0,3)
        triangle_5.GetPointIds().SetId(1,4)
        triangle_5.GetPointIds().SetId(2,42)
        triangles.InsertNextCell(triangle_5)
        # triangle 6
        triangle_6 = vtk.vtkTriangle()
        triangle_6.GetPointIds().SetId(0,4)
        triangle_6.GetPointIds().SetId(1,42)
        triangle_6.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_6)
        # triangle 7
        triangle_7 = vtk.vtkTriangle()
        triangle_7.GetPointIds().SetId(0,4)
        triangle_7.GetPointIds().SetId(1,5)
        triangle_7.GetPointIds().SetId(2,41)
        triangles.InsertNextCell(triangle_7)
        # triangle 8
        triangle_8 = vtk.vtkTriangle()
        triangle_8.GetPointIds().SetId(0,5)
        triangle_8.GetPointIds().SetId(1,41)
        triangle_8.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_8)
        # triangle 9
        triangle_9 = vtk.vtkTriangle()
        triangle_9.GetPointIds().SetId(0,5)
        triangle_9.GetPointIds().SetId(1,6)
        triangle_9.GetPointIds().SetId(2,40)
        triangles.InsertNextCell(triangle_9)
        # triangle 10
        triangle_10 = vtk.vtkTriangle()
        triangle_10.GetPointIds().SetId(0,6)
        triangle_10.GetPointIds().SetId(1,40)
        triangle_10.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_10)
        # triangle 11
        triangle_11 = vtk.vtkTriangle()
        triangle_11.GetPointIds().SetId(0,6)
        triangle_11.GetPointIds().SetId(1,7)
        triangle_11.GetPointIds().SetId(2,39)
        triangles.InsertNextCell(triangle_11)
        # triangle 12
        triangle_12 = vtk.vtkTriangle()
        triangle_12.GetPointIds().SetId(0,7)
        triangle_12.GetPointIds().SetId(1,39)
        triangle_12.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_12)
        # triangle 13
        triangle_13 = vtk.vtkTriangle()
        triangle_13.GetPointIds().SetId(0,7)
        triangle_13.GetPointIds().SetId(1,8)
        triangle_13.GetPointIds().SetId(2,38)
        triangles.InsertNextCell(triangle_13)
        # triangle 14
        triangle_14 = vtk.vtkTriangle()
        triangle_14.GetPointIds().SetId(0,8)
        triangle_14.GetPointIds().SetId(1,38)
        triangle_14.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_14)
        # triangle 15
        triangle_15 = vtk.vtkTriangle()
        triangle_15.GetPointIds().SetId(0,8)
        triangle_15.GetPointIds().SetId(1,9)
        triangle_15.GetPointIds().SetId(2,37)
        triangles.InsertNextCell(triangle_15)
        # triangle 16
        triangle_16 = vtk.vtkTriangle()
        triangle_16.GetPointIds().SetId(0,9)
        triangle_16.GetPointIds().SetId(1,37)
        triangle_16.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_16)
        # triangle 17
        triangle_17 = vtk.vtkTriangle()
        triangle_17.GetPointIds().SetId(0,9)
        triangle_17.GetPointIds().SetId(1,10)
        triangle_17.GetPointIds().SetId(2,36)
        triangles.InsertNextCell(triangle_17)
        # triangle 18
        triangle_18 = vtk.vtkTriangle()
        triangle_18.GetPointIds().SetId(0,10)
        triangle_18.GetPointIds().SetId(1,36)
        triangle_18.GetPointIds().SetId(2,35)
        triangles.InsertNextCell(triangle_18)
        # triangle 19
        triangle_19 = vtk.vtkTriangle()
        triangle_19.GetPointIds().SetId(0,10)
        triangle_19.GetPointIds().SetId(1,11)
        triangle_19.GetPointIds().SetId(2,35)
        triangles.InsertNextCell(triangle_19)
        # triangle 20
        triangle_20 = vtk.vtkTriangle()
        triangle_20.GetPointIds().SetId(0,11)
        triangle_20.GetPointIds().SetId(1,35)
        triangle_20.GetPointIds().SetId(2,34)
        triangles.InsertNextCell(triangle_20)
        # triangle 21
        triangle_21 = vtk.vtkTriangle()
        triangle_21.GetPointIds().SetId(0,11)
        triangle_21.GetPointIds().SetId(1,12)
        triangle_21.GetPointIds().SetId(2,34)
        triangles.InsertNextCell(triangle_21)
        # triangle 22
        triangle_22 = vtk.vtkTriangle()
        triangle_22.GetPointIds().SetId(0,12)
        triangle_22.GetPointIds().SetId(1,34)
        triangle_22.GetPointIds().SetId(2,33)
        triangles.InsertNextCell(triangle_22)
        # triangle 23
        triangle_23 = vtk.vtkTriangle()
        triangle_23.GetPointIds().SetId(0,12)
        triangle_23.GetPointIds().SetId(1,13)
        triangle_23.GetPointIds().SetId(2,33)
        triangles.InsertNextCell(triangle_23)
        # triangle 24
        triangle_24 = vtk.vtkTriangle()
        triangle_24.GetPointIds().SetId(0,13)
        triangle_24.GetPointIds().SetId(1,33)
        triangle_24.GetPointIds().SetId(2,32)
        triangles.InsertNextCell(triangle_24)
        # triangle 25
        triangle_25 = vtk.vtkTriangle()
        triangle_25.GetPointIds().SetId(0,13)
        triangle_25.GetPointIds().SetId(1,14)
        triangle_25.GetPointIds().SetId(2,32)
        triangles.InsertNextCell(triangle_25)
        # triangle 26
        triangle_26 = vtk.vtkTriangle()
        triangle_26.GetPointIds().SetId(0,14)
        triangle_26.GetPointIds().SetId(1,32)
        triangle_26.GetPointIds().SetId(2,31)
        triangles.InsertNextCell(triangle_26)
        # triangle 27
        triangle_27 = vtk.vtkTriangle()
        triangle_27.GetPointIds().SetId(0,14)
        triangle_27.GetPointIds().SetId(1,15)
        triangle_27.GetPointIds().SetId(2,31)
        triangles.InsertNextCell(triangle_27)
        # triangle 28
        triangle_28 = vtk.vtkTriangle()
        triangle_28.GetPointIds().SetId(0,15)
        triangle_28.GetPointIds().SetId(1,31)
        triangle_28.GetPointIds().SetId(2,30)
        triangles.InsertNextCell(triangle_28)
        # triangle 29
        triangle_29 = vtk.vtkTriangle()
        triangle_29.GetPointIds().SetId(0,15)
        triangle_29.GetPointIds().SetId(1,16)
        triangle_29.GetPointIds().SetId(2,30)
        triangles.InsertNextCell(triangle_29)
        # triangle 30
        triangle_30 = vtk.vtkTriangle()
        triangle_30.GetPointIds().SetId(0,16)
        triangle_30.GetPointIds().SetId(1,30)
        triangle_30.GetPointIds().SetId(2,29)
        triangles.InsertNextCell(triangle_30)
        # triangle 31
        triangle_31 = vtk.vtkTriangle()
        triangle_31.GetPointIds().SetId(0,16)
        triangle_31.GetPointIds().SetId(1,17)
        triangle_31.GetPointIds().SetId(2,29)
        triangles.InsertNextCell(triangle_31)
        # triangle 32
        triangle_32 = vtk.vtkTriangle()
        triangle_32.GetPointIds().SetId(0,17)
        triangle_32.GetPointIds().SetId(1,29)
        triangle_32.GetPointIds().SetId(2,28)
        triangles.InsertNextCell(triangle_32)
        # triangle 33
        triangle_33 = vtk.vtkTriangle()
        triangle_33.GetPointIds().SetId(0,17)
        triangle_33.GetPointIds().SetId(1,18)
        triangle_33.GetPointIds().SetId(2,28)
        triangles.InsertNextCell(triangle_33)
        # triangle 34
        triangle_34 = vtk.vtkTriangle()
        triangle_34.GetPointIds().SetId(0,18)
        triangle_34.GetPointIds().SetId(1,28)
        triangle_34.GetPointIds().SetId(2,27)
        triangles.InsertNextCell(triangle_34)
        # triangle 35
        triangle_35 = vtk.vtkTriangle()
        triangle_35.GetPointIds().SetId(0,18)
        triangle_35.GetPointIds().SetId(1,19)
        triangle_35.GetPointIds().SetId(2,27)
        triangles.InsertNextCell(triangle_35)
        # triangle 36
        triangle_36 = vtk.vtkTriangle()
        triangle_36.GetPointIds().SetId(0,19)
        triangle_36.GetPointIds().SetId(1,27)
        triangle_36.GetPointIds().SetId(2,26)
        triangles.InsertNextCell(triangle_36)
        # triangle 37
        triangle_37 = vtk.vtkTriangle()
        triangle_37.GetPointIds().SetId(0,19)
        triangle_37.GetPointIds().SetId(1,20)
        triangle_37.GetPointIds().SetId(2,26)
        triangles.InsertNextCell(triangle_37)
        # triangle 38
        triangle_38 = vtk.vtkTriangle()
        triangle_38.GetPointIds().SetId(0,20)
        triangle_38.GetPointIds().SetId(1,26)
        triangle_38.GetPointIds().SetId(2,25)
        triangles.InsertNextCell(triangle_38)
        # triangle 39
        triangle_39 = vtk.vtkTriangle()
        triangle_39.GetPointIds().SetId(0,20)
        triangle_39.GetPointIds().SetId(1,21)
        triangle_39.GetPointIds().SetId(2,25)
        triangles.InsertNextCell(triangle_39)
        # triangle 40
        triangle_40 = vtk.vtkTriangle()
        triangle_40.GetPointIds().SetId(0,21)
        triangle_40.GetPointIds().SetId(1,25)
        triangle_40.GetPointIds().SetId(2,24)
        triangles.InsertNextCell(triangle_40)
        # triangle 41
        triangle_41 = vtk.vtkTriangle()
        triangle_41.GetPointIds().SetId(0,21)
        triangle_41.GetPointIds().SetId(1,22)
        triangle_41.GetPointIds().SetId(2,24)
        triangles.InsertNextCell(triangle_41)
        # triangle 42
        triangle_42 = vtk.vtkTriangle()
        triangle_42.GetPointIds().SetId(0,22)
        triangle_42.GetPointIds().SetId(1,24)
        triangle_42.GetPointIds().SetId(2,23)
        triangles.InsertNextCell(triangle_42)

        self.mem_3 = vtk.vtkPolyData()
        self.mem_3.SetPoints(self.mem_pts_3)
        self.mem_3.SetPolys(triangles)

        Mapper = vtk.vtkPolyDataMapper()
        Mapper.SetInputData(self.mem_3)

        self.membrane_Actor_3 = vtk.vtkActor()
        self.membrane_Actor_3.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_3.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_3.ForceTranslucentOn()
        self.membrane_Actor_3.SetMapper(Mapper)

    def root_trace(self,root_pts_in,N_pts_display):
        N_root_pts = len(root_pts_in)
        if N_root_pts>1:
            self.root_pts = vtk.vtkPoints()
            self.root_pts.SetDataTypeToFloat()
            # add points:
            if N_root_pts > N_pts_display:
                for i in range(N_root_pts-N_pts_display,N_root_pts):
                    self.root_pts.InsertNextPoint(root_pts_in[i][0],root_pts_in[i][1],root_pts_in[i][2])
            else:
                for i in range(1,N_root_pts):
                    self.root_pts.InsertNextPoint(root_pts_in[i][0],root_pts_in[i][1],root_pts_in[i][2])
            # root spline
            self.root_spline = vtk.vtkParametricSpline()
            self.root_spline.SetPoints(self.root_pts)
            self.root_function_src = vtk.vtkParametricFunctionSource()
            self.root_function_src.SetParametricFunction(self.root_spline)
            self.root_function_src.SetUResolution(self.root_pts.GetNumberOfPoints())
            self.root_function_src.Update()
            # Radius interpolation
            self.root_radius_interp = vtk.vtkTupleInterpolator()
            self.root_radius_interp.SetInterpolationTypeToLinear()
            self.root_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.root_radius = vtk.vtkDoubleArray()
            N_spline = self.root_function_src.GetOutput().GetNumberOfPoints()
            self.root_radius.SetNumberOfTuples(N_spline)
            self.root_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.root_radius.SetTuple1(i, t)
            self.tubePolyData_root = vtk.vtkPolyData()
            self.tubePolyData_root = self.root_function_src.GetOutput()
            self.tubePolyData_root.GetPointData().AddArray(self.root_radius)
            self.tubePolyData_root.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_root = vtk.vtkTubeFilter()
            self.tuber_root.SetInputData(self.tubePolyData_root)
            self.tuber_root.SetNumberOfSides(6)
            self.tuber_root.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_root)
            lineMapper.SetScalarRange(self.tubePolyData_root.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_root.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            #self.tubeActor_root = vtk.vtkActor()
            self.tubeActor_root.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_root.SetMapper(tubeMapper)
        else:
            self.root_pts = vtk.vtkPoints()
            self.root_pts.SetDataTypeToFloat()
            # add dummy points:
            self.root_pts.InsertPoint(0,(0.0,0.0,0.0))
            self.root_pts.InsertPoint(1,(0.0,0.0,0.04))
            # root spline
            self.root_spline = vtk.vtkParametricSpline()
            self.root_spline.SetPoints(self.root_pts)
            self.root_function_src = vtk.vtkParametricFunctionSource()
            self.root_function_src.SetParametricFunction(self.root_spline)
            self.root_function_src.SetUResolution(self.root_pts.GetNumberOfPoints())
            self.root_function_src.Update()
            # Radius interpolation
            self.root_radius_interp = vtk.vtkTupleInterpolator()
            self.root_radius_interp.SetInterpolationTypeToLinear()
            self.root_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.root_radius = vtk.vtkDoubleArray()
            N_spline = self.root_function_src.GetOutput().GetNumberOfPoints()
            self.root_radius.SetNumberOfTuples(N_spline)
            self.root_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.root_radius.SetTuple1(i, t)
            self.tubePolyData_root = vtk.vtkPolyData()
            self.tubePolyData_root = self.root_function_src.GetOutput()
            self.tubePolyData_root.GetPointData().AddArray(self.root_radius)
            self.tubePolyData_root.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_root = vtk.vtkTubeFilter()
            self.tuber_root.SetInputData(self.tubePolyData_root)
            self.tuber_root.SetNumberOfSides(6)
            self.tuber_root.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_root)
            lineMapper.SetScalarRange(self.tubePolyData_root.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_root.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            self.tubeActor_root = vtk.vtkActor()
            self.tubeActor_root.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_root.SetMapper(tubeMapper)

    def tip_trace(self,tip_pts_in,N_pts_display):
        N_tip_pts = len(tip_pts_in)
        if N_tip_pts>1:
            self.tip_pts = vtk.vtkPoints()
            self.tip_pts.SetDataTypeToFloat()
            # add points:
            if N_tip_pts > N_pts_display:
                for i in range(N_tip_pts-N_pts_display,N_tip_pts):
                    self.tip_pts.InsertNextPoint(tip_pts_in[i][0],tip_pts_in[i][1],tip_pts_in[i][2])
            else:
                for i in range(1,N_tip_pts):
                    self.tip_pts.InsertNextPoint(tip_pts_in[i][0],tip_pts_in[i][1],tip_pts_in[i][2])
            # root spline
            self.tip_spline = vtk.vtkParametricSpline()
            self.tip_spline.SetPoints(self.tip_pts)
            self.tip_function_src = vtk.vtkParametricFunctionSource()
            self.tip_function_src.SetParametricFunction(self.tip_spline)
            self.tip_function_src.SetUResolution(self.tip_pts.GetNumberOfPoints())
            self.tip_function_src.Update()
            # Radius interpolation
            self.tip_radius_interp = vtk.vtkTupleInterpolator()
            self.tip_radius_interp.SetInterpolationTypeToLinear()
            self.tip_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.tip_radius = vtk.vtkDoubleArray()
            N_spline = self.tip_function_src.GetOutput().GetNumberOfPoints()
            self.tip_radius.SetNumberOfTuples(N_spline)
            self.tip_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.tip_radius.SetTuple1(i, t)
            self.tubePolyData_tip = vtk.vtkPolyData()
            self.tubePolyData_tip = self.tip_function_src.GetOutput()
            self.tubePolyData_tip.GetPointData().AddArray(self.tip_radius)
            self.tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_tip = vtk.vtkTubeFilter()
            self.tuber_tip.SetInputData(self.tubePolyData_tip)
            self.tuber_tip.SetNumberOfSides(6)
            self.tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_tip)
            lineMapper.SetScalarRange(self.tubePolyData_tip.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_tip.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            #self.tubeActor_tip = vtk.vtkActor()
            self.tubeActor_tip.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_tip.SetMapper(tubeMapper)
        else:
            self.tip_pts = vtk.vtkPoints()
            self.tip_pts.SetDataTypeToFloat()
            # add dummy points:
            self.tip_pts.InsertPoint(0,(0.0,0.0,0.0))
            self.tip_pts.InsertPoint(1,(0.0,0.0,0.01))
            # root spline
            self.tip_spline = vtk.vtkParametricSpline()
            self.tip_spline.SetPoints(self.tip_pts)
            self.tip_function_src = vtk.vtkParametricFunctionSource()
            self.tip_function_src.SetParametricFunction(self.tip_spline)
            self.tip_function_src.SetUResolution(self.tip_pts.GetNumberOfPoints())
            self.tip_function_src.Update()
            # Radius interpolation
            self.tip_radius_interp = vtk.vtkTupleInterpolator()
            self.tip_radius_interp.SetInterpolationTypeToLinear()
            self.tip_radius_interp.SetNumberOfComponents(1)
            # Tube radius
            self.tip_radius = vtk.vtkDoubleArray()
            N_spline = self.tip_function_src.GetOutput().GetNumberOfPoints()
            self.tip_radius.SetNumberOfTuples(N_spline)
            self.tip_radius.SetName("TubeRadius")
            tMin = 0.01
            tMax = 0.01
            for i in range(N_spline):
                t = (tMax-tMin)/(N_spline-1.0)*i+tMin
                self.tip_radius.SetTuple1(i, t)
            self.tubePolyData_tip = vtk.vtkPolyData()
            self.tubePolyData_tip = self.tip_function_src.GetOutput()
            self.tubePolyData_tip.GetPointData().AddArray(self.tip_radius)
            self.tubePolyData_tip.GetPointData().SetActiveScalars("TubeRadius")
            # Tube filter:
            self.tuber_tip = vtk.vtkTubeFilter()
            self.tuber_tip.SetInputData(self.tubePolyData_tip)
            self.tuber_tip.SetNumberOfSides(6)
            self.tuber_tip.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
            # Line Mapper
            lineMapper = vtk.vtkPolyDataMapper()
            lineMapper.SetInputData(self.tubePolyData_tip)
            lineMapper.SetScalarRange(self.tubePolyData_tip.GetScalarRange())
            # Tube Mapper
            tubeMapper = vtk.vtkPolyDataMapper()
            tubeMapper.SetInputConnection(self.tuber_tip.GetOutputPort())
            tubeMapper.ScalarVisibilityOff()
            # Line Actor
            lineActor = vtk.vtkActor()
            lineActor.SetMapper(lineMapper)
            # Tube actor
            self.tubeActor_tip = vtk.vtkActor()
            self.tubeActor_tip.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
            self.tubeActor_tip.SetMapper(tubeMapper)

    def add_tip_trace(self,renderer):
        renderer.AddActor(self.tubeActor_tip)

    def remove_tip_trace(self,renderer):
        renderer.RemoveActor(self.tubeActor_tip)

    def set_root_axes(self):
        # Add axes:
        self.root_axes = vtk.vtkAxesActor()
        #self.root_axes.SetTotalLength(1.0,1.0,1.0)
        self.root_axes.SetTotalLength(1.8,1.8,1.8)
        self.root_axes.SetXAxisLabelText('')
        self.root_axes.SetYAxisLabelText('')
        self.root_axes.SetZAxisLabelText('')
        self.root_axes.SetShaftTypeToCylinder()
        self.root_axes.SetCylinderRadius(0.005)
        self.root_axes.SetConeRadius(0.1)

    def add_actors(self,renderer):
        renderer.AddActor(self.tubeActor_L0)
        renderer.AddActor(self.tubeActor_L1)
        renderer.AddActor(self.tubeActor_L2)
        renderer.AddActor(self.tubeActor_L3)
        renderer.AddActor(self.tubeActor_L4)
        renderer.AddActor(self.tubeActor_L5)
        renderer.AddActor(self.tubeActor_C1)
        renderer.AddActor(self.tubeActor_C2)
        renderer.AddActor(self.tubeActor_C3)
        renderer.AddActor(self.tubeActor_A)
        renderer.AddActor(self.tubeActor_P)
        renderer.AddActor(self.membrane_Actor_0)
        renderer.AddActor(self.membrane_Actor_1)
        renderer.AddActor(self.membrane_Actor_2)
        renderer.AddActor(self.membrane_Actor_3)
        if self.root_trace_on:
            renderer.AddActor(self.tubeActor_root)
        if self.tip_trace_on:
            renderer.AddActor(self.tubeActor_tip)
        if self.root_axes_on:
            renderer.AddActor(self.root_axes)

    def transform_wing(self,s_in):
        q_norm = np.sqrt(pow(s_in[0],2)+pow(s_in[1],2)+pow(s_in[2],2)+pow(s_in[3],2))
        q0 = s_in[0]/q_norm
        q1 = s_in[1]/q_norm
        q2 = s_in[2]/q_norm
        q3 = s_in[3]/q_norm
        tx = s_in[4]
        ty = s_in[5]
        tz = s_in[6]
        M = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2, tx],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1, ty],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2), tz],
            [0,0,0,1]])
        b_angle = s_in[7]
        M_0 = self.convert_2_vtkMat(M)
        M1 = self.M_axis_1(M,b_angle/3.0)
        M_1 = self.convert_2_vtkMat(M1)
        M2 = self.M_axis_2(M1,b_angle/3.0)
        M_2 = self.convert_2_vtkMat(M2)
        M3 = self.M_axis_3(M2,b_angle/3.0)
        M_3 = self.convert_2_vtkMat(M3)
        self.tubeActor_L0.SetUserMatrix(M_0)
        self.tubeActor_L0.Modified()
        self.tubeActor_L1.SetUserMatrix(M_0)
        self.tubeActor_L1.Modified()
        self.tubeActor_L2.SetUserMatrix(M_0)
        self.tubeActor_L2.Modified()
        self.tubeActor_L3.SetUserMatrix(M_0)
        self.tubeActor_L3.Modified()
        self.tubeActor_C1.SetUserMatrix(M_1)
        self.tubeActor_C1.Modified()
        self.tubeActor_L4.SetUserMatrix(M_2)
        self.tubeActor_L4.Modified()
        self.tubeActor_C2.SetUserMatrix(M_2)
        self.tubeActor_C2.Modified()
        self.tubeActor_L5.SetUserMatrix(M_2)
        self.tubeActor_L5.Modified()
        self.tubeActor_C3.SetUserMatrix(M_3)
        self.tubeActor_C3.Modified()
        self.tubeActor_A.SetUserMatrix(M_1)
        self.tubeActor_A.Modified()
        self.tubeActor_P.SetUserMatrix(M_2)
        self.tubeActor_P.Modified()
        self.membrane_Actor_0.SetUserMatrix(M_0)
        self.membrane_Actor_0.Modified()
        self.membrane_Actor_1.SetUserMatrix(M_1)
        self.membrane_Actor_1.Modified()
        self.membrane_Actor_2.SetUserMatrix(M_2)
        self.membrane_Actor_2.Modified()
        self.membrane_Actor_3.SetUserMatrix(M_3)
        self.membrane_Actor_3.Modified()
        #self.renWin.Render()
        # Calculate keypoints positions:
        scale_arr = np.array([[self.scale_now],[self.scale_now],[self.scale_now],[1.0]])
        key0_pts = np.dot(M,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[0,1,2,8],:])))
        key1_pts = np.dot(M1,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[3,7,9],:])))
        key2_pts = np.dot(M2,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[4,6],:])))
        key3_pts = np.dot(M3,np.multiply(scale_arr,np.transpose(self.wing_key_pts[[5,4],:])))
        self.key_points = np.zeros((3,10))
        self.key_points[:,0] = key0_pts[0:3,0] # keypoint 0
        self.key_points[:,1] = key0_pts[0:3,1] # keypoint 1
        self.key_points[:,2] = key0_pts[0:3,2] # keypoint 2
        self.key_points[:,3] = key1_pts[0:3,0] # keypoint 3
        self.key_points[:,4] = key2_pts[0:3,0] # keypoint 4
        self.key_points[:,5] = key3_pts[0:3,0] # keypoint 5
        self.key_points[:,6] = key2_pts[0:3,1] # keypoint 6
        self.key_points[:,7] = key1_pts[0:3,1] # keypoint 7
        self.key_points[:,8] = key0_pts[0:3,3] # keypoint 8
        self.key_points[:,9] = key1_pts[0:3,2] # keypoint 9
        # Update root and wing trace:
        self.root_pts_list.append(np.array([tx,ty,tz]))
        self.root_trace(self.root_pts_list,100000)
        self.tip_pts_list.append(self.key_points[:,2])
        self.tip_trace(self.tip_pts_list,100000)
        # Update root trace:
        root_transform = vtk.vtkTransform()
        root_transform.Translate(tx,ty,tz)
        self.root_axes.SetUserTransform(root_transform);
        self.root_axes.Modified()
        return M, M1, M2, M3

    def clear_root_tip_pts(self,root_pts_in,tip_pts_in):
        self.root_pts_list = [root_pts_in,root_pts_in]
        self.tip_pts_list = [tip_pts_in,tip_pts_in]
        self.root_trace(self.root_pts_list,100000)
        self.tip_trace(self.tip_pts_list,100000)

    def convert_2_vtkMat(self,M):
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
        return M_vtk

    def M_axis_1(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.0
        q2 = 1.0*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        M_out[3,3] = 1.0
        return M_out

    def M_axis_2(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.05959*np.sin(b_angle/2.0)
        q2 = 0.99822*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        TA = np.dot(R_out,np.array([-0.0867,-0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,-0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        M_out[3,3] = 1.0
        return M_out

    def M_axis_3(self,M_in,b_angle):
        q0 = np.cos(b_angle/2.0)
        q1 = 0.36186*np.sin(b_angle/2.0)
        q2 = 0.93223*np.sin(b_angle/2.0)
        q3 = 0.0
        q_norm = np.sqrt(pow(q0,2)+pow(q1,2)+pow(q2,2)+pow(q3,2))
        q0 /= q_norm
        q1 /= q_norm
        q2 /= q_norm
        q3 /= q_norm
        R = np.array([[2*pow(q0,2)-1+2*pow(q1,2), 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
            [2*q1*q2-2*q0*q3, 2*pow(q0,2)-1+2*pow(q2,2), 2*q2*q3+2*q0*q1],
            [2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 2*pow(q0,2)-1+2*pow(q3,2)]])
        R_in = M_in[0:3,0:3]
        R_out = np.squeeze(np.dot(R_in,R))
        M_out = np.zeros((4,4))
        M_out[0:3,0:3] = R_out
        M_out[0:3,3] = M_in[0:3,3]
        TA = np.dot(R_out,np.array([-0.0867,-0.0145,0.0]))
        TB = np.dot(R_in,np.array([-0.0867,-0.0145,0.0]))
        M_out[0:3,3] -= TA-TB
        M_out[3,3] = 1.0
        return M_out

    def scale_wing(self,scale_in):
        self.scale_now = scale_in
        self.tubeActor_L0.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L0.Modified()
        self.tubeActor_L1.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L1.Modified()
        self.tubeActor_L2.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L2.Modified()
        self.tubeActor_L3.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L3.Modified()
        self.tubeActor_L4.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L4.Modified()
        self.tubeActor_L5.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_L5.Modified()
        self.tubeActor_C1.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_C1.Modified()
        self.tubeActor_C2.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_C2.Modified()
        self.tubeActor_C3.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_C3.Modified()
        self.tubeActor_A.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_A.Modified()
        self.tubeActor_P.SetScale(scale_in,scale_in,scale_in)
        self.tubeActor_P.Modified()
        self.membrane_Actor_0.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_0.Modified()
        self.membrane_Actor_1.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_1.Modified()
        self.membrane_Actor_2.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_2.Modified()
        self.membrane_Actor_3.SetScale(scale_in,scale_in,scale_in)
        self.membrane_Actor_3.Modified()

    def return_polydata_mem0(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_0)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def return_polydata_mem1(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_1)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def return_polydata_mem2(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_2)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def return_polydata_mem3(self):
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_now,self.scale_now,self.scale_now)
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_3)
        transformPD.Update()
        poly_out = transformPD.GetOutput()
        return poly_out

    def write_stl(self,save_dir,scale_in):
        transform = vtk.vtkTransform()
        transform.Scale(scale_in,scale_in,scale_in)
        stlWriter = vtk.vtkSTLWriter()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_0)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_0_R.stl'))
        stlWriter.Write()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_1)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_1_R.stl'))
        stlWriter.Write()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_2)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_2_R.stl'))
        stlWriter.Write()
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputData(self.mem_3)
        transformPD.Update()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(transformPD.GetOutputPort())
        triangle_filter.Update()
        stlWriter.SetInputConnection(triangle_filter.GetOutputPort())
        stlWriter.SetFileName(os.path.join(save_dir, 'membrane_3_R.stl'))
        stlWriter.Write()

    def set_Color(self,color_vec):
        self.vein_clr = color_vec[0]
        self.mem_clr = color_vec[1]
        self.mem_opacity = color_vec[2]
        self.tubeActor_L0.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L0.Modified()
        self.tubeActor_L1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L1.Modified()
        self.tubeActor_L2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L2.Modified()
        self.tubeActor_L3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L3.Modified()
        self.tubeActor_L4.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L4.Modified()
        self.tubeActor_L5.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_L5.Modified()
        self.tubeActor_C1.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C1.Modified()
        self.tubeActor_C2.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C2.Modified()
        self.tubeActor_C3.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_C3.Modified()
        self.tubeActor_A.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_A.Modified()
        self.tubeActor_P.GetProperty().SetColor(self.vein_clr[0],self.vein_clr[1],self.vein_clr[2])
        self.tubeActor_P.Modified()
        self.membrane_Actor_0.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_0.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_0.ForceTranslucentOn()
        self.membrane_Actor_0.Modified()
        self.membrane_Actor_1.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_1.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_1.ForceTranslucentOn()
        self.membrane_Actor_1.Modified()
        self.membrane_Actor_2.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_2.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_2.ForceTranslucentOn()
        self.membrane_Actor_2.Modified()
        self.membrane_Actor_3.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.membrane_Actor_3.GetProperty().SetOpacity(self.mem_opacity)
        self.membrane_Actor_3.ForceTranslucentOn()
        self.membrane_Actor_3.Modified()

    def visualize(self):
        self.ren = vtk.vtkRenderer()
        self.ren.SetUseDepthPeeling(True)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.ren.AddActor(self.tubeActor_L0)
        self.ren.AddActor(self.tubeActor_L1)
        self.ren.AddActor(self.tubeActor_L2)
        self.ren.AddActor(self.tubeActor_L3)
        self.ren.AddActor(self.tubeActor_L4)
        self.ren.AddActor(self.tubeActor_L5)
        self.ren.AddActor(self.tubeActor_C1)
        self.ren.AddActor(self.tubeActor_C2)
        self.ren.AddActor(self.tubeActor_C3)
        self.ren.AddActor(self.tubeActor_A)
        self.ren.AddActor(self.tubeActor_P)
        self.ren.AddActor(self.membrane_Actor_0)
        self.ren.AddActor(self.membrane_Actor_1)
        self.ren.AddActor(self.membrane_Actor_2)
        self.ren.AddActor(self.membrane_Actor_3)
        self.ren.SetBackground(1.0,1.0,1.0)
        self.renWin.SetSize(1200,1200)
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def plotKeyPoints(self,renderer,y_in,clr):
        # plot keypoints in 3D
        N_pts = 10
        for i in range(N_pts):
            key_sphere = vtk.vtkSphereSource()
            key_sphere.SetCenter(y_in[i*3],y_in[i*3+1],y_in[i*3+2])
            key_sphere.SetRadius(0.05)
            key_mapper = vtk.vtkPolyDataMapper()
            key_mapper.SetInputConnection(key_sphere.GetOutputPort())
            key_actor = vtk.vtkActor()
            key_actor.SetMapper(key_mapper)
            key_actor.GetProperty().SetColor(clr[0],clr[1],clr[2])
            renderer.AddActor(key_actor)

class StrokePlane():

    def __init__(self):
        self.mem_clr = (0.01,0.01,0.01)
        self.mem_opacity = 0.3
        self.joint_L = np.array([0.0,0.7,0.0])
        self.joint_R = np.array([0.0,-0.7,0.0])
        self.radius = 3.0
        self.SRF_planes()

    def set_joint_loc(self,joint_loc_L,joint_loc_R):
        self.joint_L = joint_loc_L
        self.joint_R = joint_loc_R

    def set_SRF_radius(self,radius_in):
        self.radius = radius_in

    def SRF_planes(self):
        N_pts = 50
        self.SRF_L_pts = vtk.vtkPoints()
        self.SRF_L_pts.SetDataTypeToFloat()
        self.SRF_R_pts = vtk.vtkPoints()
        self.SRF_R_pts.SetDataTypeToFloat()
        theta_range = np.linspace(-np.pi/2.0,np.pi/2.0,num=N_pts,endpoint=True)
        self.SRF_L_pts.InsertPoint(0,(self.joint_L[0],self.joint_L[1],self.joint_L[2]))
        self.SRF_R_pts.InsertPoint(0,(self.joint_R[0],self.joint_R[1],self.joint_R[2]))
        for i in range(N_pts):
            self.SRF_L_pts.InsertPoint(i+1,(self.joint_L[0]+self.radius*np.sin(theta_range[i]),self.joint_L[1]+self.radius*np.cos(theta_range[i]),self.joint_L[2]))
            self.SRF_R_pts.InsertPoint(i+1,(self.joint_R[0]-self.radius*np.sin(theta_range[i]),self.joint_R[1]-self.radius*np.cos(theta_range[i]),self.joint_R[2]))
        triangles = vtk.vtkCellArray()
        N_triangles = N_pts-1
        for j in range(N_triangles):
            triangle_j = vtk.vtkTriangle()
            triangle_j.GetPointIds().SetId(0,0)
            triangle_j.GetPointIds().SetId(1,j+1)
            triangle_j.GetPointIds().SetId(2,j+2)
            triangles.InsertNextCell(triangle_j)
        self.SRF_L = vtk.vtkPolyData()
        self.SRF_L.SetPoints(self.SRF_L_pts)
        self.SRF_L.SetPolys(triangles)
        self.SRF_R = vtk.vtkPolyData()
        self.SRF_R.SetPoints(self.SRF_R_pts)
        self.SRF_R.SetPolys(triangles)
        Mapper_L = vtk.vtkPolyDataMapper()
        Mapper_L.SetInputData(self.SRF_L)
        Mapper_R = vtk.vtkPolyDataMapper()
        Mapper_R.SetInputData(self.SRF_R)
        self.SRF_Actor_L = vtk.vtkActor()
        self.SRF_Actor_L.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.SRF_Actor_L.GetProperty().SetOpacity(self.mem_opacity)
        self.SRF_Actor_L.ForceTranslucentOn()
        self.SRF_Actor_L.SetMapper(Mapper_L)
        self.SRF_Actor_R = vtk.vtkActor()
        self.SRF_Actor_R.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.SRF_Actor_R.GetProperty().SetOpacity(self.mem_opacity)
        self.SRF_Actor_R.ForceTranslucentOn()
        self.SRF_Actor_R.SetMapper(Mapper_R)

    def add_actors(self,renderer):
        renderer.AddActor(self.SRF_Actor_L)
        renderer.AddActor(self.SRF_Actor_R)

    def set_Color(self,color_vec):
        self.mem_clr = color_vec[0]
        self.mem_opacity = color_vec[1]
        self.SRF_Actor_L.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.SRF_Actor_L.Modified()
        self.SRF_Actor_R.GetProperty().SetColor(self.mem_clr[0],self.mem_clr[1],self.mem_clr[2])
        self.SRF_Actor_R.Modified()

    def scale_SRF(self,scale_in):
        self.scale_now = scale_in
        self.SRF_Actor_L.SetScale(scale_in,scale_in,scale_in)
        self.SRF_Actor_L.Modified()
        self.SRF_Actor_R.SetScale(scale_in,scale_in,scale_in)
        self.SRF_Actor_R.Modified()

# -------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    wing = WingModel_R()
    s_start = np.array([np.cos(np.pi/6.0),0.0,np.sin(np.pi/6.0),0.0,0.0,0.5,0.0,-np.pi/4.0])
    wing.scale_wing(0.85)
    wing.transform_wing(s_start)
    wing.visualize()
