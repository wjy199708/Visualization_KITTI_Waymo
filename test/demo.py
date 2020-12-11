import numpy
from mayavi import mlab


# data
px=numpy.arange(0,10000,1)
py=numpy.arange(0,50000,5)
pz=numpy.zeros_like(px)
s=0.5
# render
pts=mlab.points3d(px,py,pz)
T_max = len(px)
delayer=40
@mlab.animate(delay=delayer)
def anim_loc():
    for i in numpy.arange(2, T_max,50):
        _x = px[0:i]
        _y = px[0:i]
        _z = pz[0:i]
        pts.mlab_source.reset( x = _x, y = _y, z = _z, )
        yield

anim_loc()
mlab.show()