import sys
sys.path.append('../..')

from paraview.simple import *

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def plot_paraview(filepath_pvd, filepath_figure, cbar_RGB):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'PVD Reader'
    data = PVDReader(FileName=filepath_pvd)

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    renderView1.OrientationAxesVisibility = 0

    # show data in view
    data_pvdDisplay = Show(data, renderView1)

    #=== Colourbar ===#
    f_677LUT = GetColorTransferFunction('f_677')
    f_677LUT.RGBPoints = cbar_RGB
    f_677LUT.ScalarRangeInitialized = 1.0
    f_677LUTColorBar = GetScalarBar(f_677LUT, renderView1)
    f_677LUTColorBar.Title = ''
    f_677LUTColorBar.ComponentTitle = ''
    f_677LUTColorBar.WindowLocation = 'AnyLocation'
    f_677LUTColorBar.Position = [0.7915282392026579, 0.10885167464114837]
    f_677LUTColorBar.ScalarBarLength = 0.38741626794258327
    f_677LUTColorBar.ScalarBarThickness = 5
    f_677LUTColorBar.LabelFontSize = 5
    f_677LUTColorBar.LabelColor = [0.0, 0.0, 0.0]

    # current camera placement for renderView1
    renderView1.CameraPosition = [-4.682222107454641, 0.693060198956223, 4.73086391559853]
    renderView1.CameraViewUp = [0.08329790299678129, 0.9945152399476537, -0.06325264317164714]
    renderView1.CameraParallelScale = 1.7320508075688772

    # save screenshot
    SaveScreenshot(filepath_figure, renderView1, ImageResolution=[1432, 793], TransparentBackground=1)
