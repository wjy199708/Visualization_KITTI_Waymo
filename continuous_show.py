import os
import numpy as np
import struct
import open3d as o3d
import open3d

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def main():
    root_dir=os.path.join(os.getcwd(),'data/object/training/velodyne')
    filename=os.listdir(root_dir)
    file_number=len(filename)

    pcd=open3d.open3d.geometry.PointCloud()

    for i in range(file_number):
        path=os.path.join(root_dir, filename[i])
        print(path)
        example=read_bin_velodyne(path)
        # From numpy to Open3D
        pcd.points= open3d.open3d.utility.Vector3dVector(example)
        open3d.open3d.visualization.draw_geometries([pcd])
from glob import glob
def continuous_show_3d():
    root_dir=os.path.join(os.getcwd(),'data/waymo/visualization/')
    files = os.listdir(root_dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pointcloud = o3d.geometry.PointCloud()
    to_reset = True
    vis.add_geometry(pointcloud)
    for f in glob(os.path.join(root_dir,'*.bin')):
        pcd = read_bin_velodyne(root_dir + os.path.basename(f))   #此处读取的pcd文件,也可读取其他格式的
        # pcd = np.asarray(pcd.points).reshape((-1, 3))
        pointcloud.points = o3d.utility.Vector3dVector(pcd)  # 如果使用numpy数组可省略上两行
        vis.update_geometry()
        if to_reset:
            vis.reset_view_point(True)
            to_reset = False
        vis.poll_events()
        vis.update_renderer()
if __name__=="__main__":
    # main()
    continuous_show_3d()