import numpy as np
import mayavi
import mayavi.mlab as mlab
import argparse
from glob import glob
import os
import time
# from tvtk.tools import visual


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--index',
                       type=int,
                       help='the index file of bin that you wanna show ',
                       required=True)
    parse.add_argument(
        '--waymo',
        help='if using waymo dataset',
        action='store_true',
        required=True,
    )
    parse.add_argument(
        '--continuous',
        action='store_true',
        help=
        'show much more frames lidar datas,if it not given then you will get one frame scence point cloud'
    )
    args = parse.parse_args()
    return args


def get_lidar_xyzrd(base_dir, index):
    ind = index
    try:
        pointcloud = np.fromfile('{}/{}.bin'.format(base_dir,
                                                    str('%06d' % (ind))),
                                 dtype=np.float32,
                                 count=-1).reshape([-1, 4])
    except ValueError:
        pointcloud = np.fromfile('{}/{}.bin'.format(base_dir,
                                                    str('%06d' % (ind))),
                                 dtype=np.float64,
                                 count=-1).reshape([-1, 4])
    else:
        pointcloud = np.fromfile('{}/{}.bin'.format(base_dir,
                                                    str('%06d' % (ind))),
                                 dtype=np.float32,
                                 count=-1).reshape([-1, 4])
    print(pointcloud.shape)

    print(pointcloud.shape)
    x = pointcloud[:, 0]  # x position of point
    y = pointcloud[:, 1]  # y position of point
    z = pointcloud[:, 2]  # z position of point
    r = pointcloud[:, 3]  # reflectance value of point
    d = np.sqrt(x**2 + y**2)  # Map Distance from sensor
    return x, y, z, r, d


def show_one_frame_lidar(index, base_dir):
    ind = index
    # def LIDAR_show(binData):
    x, y, z, r, d = get_lidar_xyzrd(base_dir, index)

    vals = 'height'
    if vals == "height":
        col = z  #威力等雷达在采集车上的方向是x向前y向左z向上
    else:
        col = d

    scalars = [1.5, 1.5]
    #根据在这一时刻采集到的数据，将这一帧的数据画出来，由于是3D的数据，所以使用mayavi来画出3D数据图
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    # visual.set_viewer(fig)
    mayavi.mlab.points3d(
        x,
        y,
        z,
        col,  # Values used for Color
        mode="point",  #sphere point
    )
    # colormap='spectral',  # 'bone', 'copper', 'gnuplot','spectral'
    # # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
    # figure=fig,
    # scale_factor=0.05)

    # mayavi.mlab.plot3d()
    x = np.linspace(5, 5, 50)
    y = np.linspace(0, 0, 50)
    z = np.linspace(0, 5, 50)
    mayavi.mlab.plot3d(x, y, z)
    mayavi.mlab.show()


def show_more_frames_lidar(continuous_lidar_dir):
    index_lists = [
        int(os.path.basename(x).replace('.bin', ''))
        for x in glob(os.path.join(continuous_lidar_dir, '*.bin'))
    ]

    # print(index_lists)
    first_frame = (x, y, z, r, d) = get_lidar_xyzrd(continuous_lidar_dir,
                                                    index_lists[0])
    vals = 'height'
    if vals == "height":
        col = z  #威力等雷达在采集车上的方向是x向前y向左z向上
    else:
        col = d
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    frame_lidar = mlab.points3d(
        first_frame[0],
        first_frame[1],
        first_frame[2],
        # col,  # Values used for Color
        mode="point",  #sphere point
        # colormap='spectral',  # 'bone', 'copper', 'gnuplot','spectral'
        # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
        # figure=fig,
        # scale_factor=0.05)
    )

    @mlab.animate(delay=20)
    def lidar_animate():
        f = mlab.gcf()
        while True:
            for index in index_lists[1:]:
                # time.sleep(0.5)
                print('-*' * 20)
                print('当前为{}'.format(index))
                print('Updating scene...')
                print('-*' * 20)
                x1, y1, z1, r1, d1 = get_lidar_xyzrd(continuous_lidar_dir,
                                                     index)
                frame_lidar.mlab_source.reset(x=x1, y=y1, z=z1)
                f.scene.render()
                yield

    lidar_animate()
    time.sleep(2)
    mlab.show()


def main():
    args = parse_args()
    base_dir = '../data/object/training/velodyne' if not args.waymo else '../data/waymo/visualization'
    if args.continuous:
        continuous_lidar_dir = base_dir

        show_more_frames_lidar(continuous_lidar_dir)
        # mlab.show()
    else:
        show_one_frame_lidar(args.index, base_dir)


if __name__ == '__main__':
    main()
