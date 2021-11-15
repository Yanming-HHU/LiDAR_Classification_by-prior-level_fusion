"""Feature fusion
get feature from raster data 

Xiaoqiang Liu      2019/07/17
"""

import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *

def get_rowcol(geo_point, geo_raster):
    tmp_1 = np.array([[geo_point[0]-geo_raster[0], geo_raster[2]],
                        [geo_point[1]-geo_raster[3], geo_raster[5]]])
    tmp_2 = np.array([[geo_raster[1], geo_raster[2]],
                        [geo_raster[4], geo_raster[5]]])
    row = np.linalg.det(tmp_1) / np.linalg.det(tmp_2)
    
    tmp_1 = np.array([[geo_point[0]-geo_raster[0], geo_raster[1]],
                        [geo_point[1]-geo_raster[3], geo_raster[4]]])
    tmp_2 = np.array([[geo_raster[2], geo_raster[1]],
                        [geo_raster[5], geo_raster[4]]])
    col = np.linalg.det(tmp_1) / np.linalg.det(tmp_2)
    
    return np.int(row), np.int(col)

def get_geo(pixel, geo_raster):
    Xgeo = geo_raster[0] + pixel[0]*geo_raster[1] + pixel[1]*geo_raster[2]
    Ygeo = geo_raster[3] + pixel[0]*geo_raster[4] + pixel[1]*geo_raster[5]
    return [Xgeo, Ygeo]



def feature_from_raster(point_cloud, raster_path):
    """ get additional point cloud features from corresponding raster
    Args:
        point_cloud: np.array([n,3]), point cloud data
        raster_path: str, path to raster file    
    """
    try:
        raster_data = gdal.Open(raster_path, gdal.GA_ReadOnly)
    except IOError:
        print('Error: cannot open file:{}'.format(raster_path))
    
    feature = []
    GeoTransform = raster_data.GetGeoTransform()
    for i in range(point_cloud.shape[0]):
        point_xy = point_cloud[i][0:2]
        row, col = get_rowcol(point_xy, GeoTransform)
        pixel_value = raster_data.ReadAsArray(xoff=row, yoff=col, xsize=1, ysize=1)
        pixel_value = list(pixel_value.flatten())
        feature += pixel_value
        
    feature = np.array(feature)
    feature = feature.reshape([-1,raster_data.RasterCount])
    return feature


if __name__ == '__main__':
    
    data = np.loadtxt('./Vaihingen3D_Training.txt')
    raster_path = './train.tif'
    feature_from_raster = feature_from_raster(data[:, 0:3], raster_path)
    tmp = np.hstack([data[:, 0:3], feature_from_raster])
    np.savetxt('tmp.txt', tmp, fmt='%.2f,%.2f,%.2f,%d,%d,%d')