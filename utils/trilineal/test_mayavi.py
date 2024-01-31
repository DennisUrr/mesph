"""

Purpose: Generating cubic data from FARGO3D, 
         using the trilinear module and ploting 
         it with mayavi. At the end, it is also 
         saved an x3d file.

"""

from mayavi import mlab
import numpy as np
import trilinear as t

def extend_field(field):
    extended_field = np.ndarray([field.shape[0],field.shape[1],field.shape[2]+1])
    extended_field[:,:,:-1] = field
    extended_field[:,:,-1]  = field[:,:,0]
    return extended_field

# directory = "/data/sims_perez/diskhydro.das.uchile.cl/dump/sims/planetvortex_p10J/"
directory = "/Users/seba/Research/HD100546_2019/sims_planetvortex_p10J/"

#Note: Data has to be inside the interval [-pi,pi]
phi   = np.loadtxt(directory+"domain_x.dat")[:]-np.pi
r     = np.loadtxt(directory+"domain_y.dat")[3:-3]
theta = np.loadtxt(directory+"domain_z.dat")[3:-3]

rmed        = 0.5*(r[1:]+r[:-1])
thetamed    = 0.5*(theta[1:]+theta[:-1])
phimed      = np.ndarray(phi.shape)
phimed[:-1] = 0.5*(phi[1:]+phi[:-1])
phimed[-1]  = phimed[-2] + (phi[1]-phi[0])

#Cartesian mesh
nxc = 400
nyc = 400
nzc = 40

xc = np.linspace(-r.max(),r.max(),nxc)
yc = np.linspace(-r.max(),r.max(),nyc)
zc = np.linspace(r.max()*np.cos(thetamed.max()),r.max()*np.cos(thetamed.min()),nzc)

YC,ZC,XC = np.meshgrid(yc,zc,xc)

#I remove one element from r, theta
r = r[:-1]
theta = theta[:-1]

#if this is true, we use the trilinear module to generate the data
generate_files = False

#---------------------------------------------------------------------

if (generate_files):
    
    field_ext = extend_field(np.log10(np.fromfile(directory+"gasdens10000.dat").reshape(theta.shape[0],r.shape[0],phi.shape[0]-1)))
    density = t.trilinear(field_ext,rmed,phimed,thetamed,xc,yc,zc,field_ext.min())
    np.save("density",density)
    
    # field0 = np.fromfile(directory+"gasvx0.dat").reshape(theta.shape[0],r.shape[0],phi.shape[0]-1)
    field  = np.fromfile(directory+"gasvx10000.dat").reshape(theta.shape[0],r.shape[0],phi.shape[0]-1)
    # field -= field0 #this is to get the vortex velocity with respect to the keplerian flow
    field_ext = extend_field(field)
    vphi = t.trilinear(field_ext,rmed,phi,thetamed,xc,yc,zc,0.0)
    
    field_ext = extend_field(np.fromfile(directory+"gasvy10000.dat").reshape(theta.shape[0],r.shape[0],phi.shape[0]-1))
    vrad = t.trilinear(field_ext,r,phimed,thetamed,xc,yc,zc,0.0)
    
    field_ext = extend_field(np.fromfile(directory+"gasvz10000.dat").reshape(theta.shape[0],r.shape[0],phi.shape[0]-1))
    vtheta = t.trilinear(field_ext,rmed,phimed,theta,xc,yc,zc,0.0)
    
    #Creating the cartesian velocity
    
    thetac = np.arctan2(np.sqrt(XC**2+YC**2),ZC)
    rc     = np.sqrt(XC**2+YC**2+ZC**2);
    phic   = np.arctan2(YC,XC);	

    VX = vrad
    VY = vphi
    VZ = vtheta
    # VX = - vphi*np.sin(phic)   + vrad*np.cos(phic)*np.sin(thetac) + vtheta*np.cos(phic)*np.cos(thetac);
    # VY =   vphi*np.cos(phic)   + vrad*np.sin(phic)*np.sin(thetac) + vtheta*np.sin(phic)*np.cos(thetac);
    # VZ =   vrad*np.cos(thetac) + vtheta*np.sin(thetac);

    np.save("vx",VX)
    np.save("vy",VY)
    np.save("vz",VZ)

    exit()
    
else:
     
    RHO = np.load("density.npy")
    VX  = np.load("vx.npy")
    VY  = np.load("vy.npy")
    VZ  = np.load("vz.npy")

# print(VZ.min()
# print(VZ.max()
#Mayavi figure
figure = mlab.figure(size=(1280,720))
 
#these variables are used to put the data in a physical domain
deltas = np.array([xc[1]-xc[0],yc[1]-yc[0],zc[1]-zc[0]])
origin = np.array((xc.min(),yc.min(),zc.min()))

#Generating the scalar_field (gas density) -----------------
# 
# src = mlab.pipeline.scalar_field(RHO.T)
src = mlab.pipeline.scalar_field(VX.T)
# the above call makes a copy of the array, so we delete it
del RHO
# del VZ
# These two lines below specify the domain
src.spacing = deltas
src.origin = origin
#-----------------------------------------------------------

#Sliders ---------------------------------------------------
mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', slice_index=0)
mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', slice_index=xc.shape[0]/4)
src.children[0].scalar_lut_manager.data_range = np.array([-VZ.max(),VZ.max()])
src.children[0].scalar_lut_manager.lut_mode = u'magma'
#-----------------------------------------------------------

# Axes properties ------------------------------------------
# axes = mlab.axes()
# axes.label_text_property.bold   = False
# axes.label_text_property.italic = False
# axes.title_text_property.bold   = False
# axes.title_text_property.italic = False
# axes.axes.number_of_labels      = 10
# axes.axes.y_axis_visibility     = False
# axes.axes.label_format          = '%-#2.1f'
#------------------------------------------------------------

# Generating the vector field -------------------------------
vector_field = mlab.pipeline.vector_field(VX.T, VY.T, VZ.T)
# the above call makes a copy of the arrays, so we delete them
del VX, VY, VZ
vector_field.spacing = deltas
vector_field.origin = origin
magnitude = mlab.pipeline.extract_vector_norm(vector_field)
#------------------------------------------------------------

#STREAMLINES FROM A POINT SOURCE
streamlines = []
nstreams = 30
x =  1.3 + 0.5*np.random.rand(nstreams)
y = -1.7 + 0.5*np.random.rand(nstreams)
z = np.random.rand(nstreams)*0.5

for x0,y0,z0 in zip(x,y,z):
    streamline = mlab.pipeline.streamline(magnitude,
                                          seedtype='point',
                                          integration_direction='forward',
                                          #integration_direction='both',
                                          colormap='jet',
                                          # colormap='viridis',
                                          seed_visible=False,
                                          reset_zoom=False)
    streamline.seed.widget.position = np.array([x0,y0,z0])
    streamlines.append(streamline)

x =  -0.8-0.4*np.random.rand(nstreams)
y =  -0.2+0.4*np.random.rand(nstreams)
z = np.random.rand(nstreams)*0.5
    
#We now modify the colormap
# magnitude.children[0].scalar_lut_manager.data_range = np.array([0,0.1])
#mlab.outline(magnitude)

#We finally update the camera position ------------------------------
scene = mlab.get_engine().scenes[0]
scene.scene.isometric_view()
scene.scene.background            = (0.0, 0.0, 0.0)
scene.scene.camera.position       = [6.0, 4.0, 4.0]
scene.scene.camera.focal_point    = [0.17, 0.0, 0.3]
scene.scene.camera.view_angle     = 40.0
scene.scene.camera.view_up        = [-0.3, -0.15, 0.95]
scene.scene.camera.clipping_range = [0.02, 22.0]
#------------------------------------------------------------

# mlab.savefig("data/output.wrl")
# mlab.savefig("data/output.x3d")
# mlab.savefig("data/output.jpg")
mlab.show()
