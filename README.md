# 3D urban land cover classification by prior-level fusion of LiDAR point cloud and optical imagery

The heterogeneity of urban landscape in the vertical direction should not be neglected in urban ecology research, which required the urban land cover product transformation from two-dimensional to three-dimensional by using LiDAR point clouds. Previous studies have demonstrated that the performance of two-dimensional land cover classification can be improved by fusing optical imagery and LiDAR data using several strategies. However, the fusion of Li-DAR point clouds and optical imagery for three-dimensional land cover classification is rarely studied, especially under the popular deep learning framework. 

In this research, we proposed a novel prior-level fusion strategy and compared it with the no-fusion strategy (baseline) and other three commonly used fusion strategies including point-level, feature-level, and decision-level. The proposed prior-level fusion strategy first referred to the two-dimensional land cover derived from optical imagery as the prior for three-dimensional classification. Then, the LiDAR point cloud is linked to the prior through the nearest neighbor method and classified by a deep neural network. 

The proposed prior-fusion strategy has higher overall accuracy (82.47%) on data from the International Society for Photogrammetry and Remote Sensing, compare to baseline (74.62%), point-level (79.86%), feature-level (76.22%), and decision-level (81.12%). 

The improved accuracy was from that: 
(1) Fusing optical imagery to LiDAR point clouds can improve the performance of three-dimensional urban land cover classification.
(2) The proposed prior-level strategy directly used the semantic information provided by the two-dimension land cover classification rather than the original spectral information of optical imagery. 
(3) The proposed prior-level fusion strategy is a series form that may fill the gap between two-dimensional and three-dimensional landcover classification.
