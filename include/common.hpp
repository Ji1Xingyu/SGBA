//
// Created by xingyu on 31/8/23.
//

#ifndef FAST_LIO_COMMON_HPP
#define FAST_LIO_COMMON_HPP

#define PCL_NO_PRECOMPILE // !! BEFORE ANY PCL INCLUDE!!

#include <iostream>                 // Standard I/O operations
#include <fstream>                  // File stream operations
#include <vector>                   // Standard vector container
#include <pcl/point_cloud.h>        // PCL point cloud structure
#include <pcl/io/pcd_io.h>          // PCL PCD file I/O#include <iostream>                 // Standard I/O operations
#include <vector>                   // Standard vector container
#include <pcl/point_cloud.h>        // PCL point cloud structure
#include <pcl/io/pcd_io.h>          // PCL PCD file I/O

#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/impl/instantiate.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/ThreadPool>
#include <unsupported/Eigen/CXX11/Tensor>

struct EIGEN_ALIGN16 PointXYZLIS {
    PCL_ADD_POINT4D; // Inherit XYZ coordinates
    float intensity;
    std::uint16_t label;
    std::uint32_t seq;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; // Ensure proper alignment

// Register the custom point type with PCL macros
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZLIS,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, intensity, intensity)
                                          (std::uint16_t, label, label)
                                          (std::uint32_t, seq, id)
)

PCL_INSTANTIATE(KdTreeFLANN, PointXYZLIS)

struct EIGEN_ALIGN16 PointXYZCLS {
    PCL_ADD_POINT4D; // Inherit XYZ coordinates
    std::uint16_t cluster;
    std::uint16_t label;
    std::uint32_t seq;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}; // Ensure proper alignment

// Register the custom point type with PCL macros
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZCLS,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (std::uint16_t, cluster, cluster)
                                          (std::uint16_t, label, label)
                                          (std::uint32_t, seq, id)
)

PCL_INSTANTIATE(KdTreeFLANN, PointXYZCLS)

class Vector3iHash {
public:
    size_t operator()(const Eigen::Vector3i& x) const {
        size_t seed = 0;
        boost::hash_combine(seed, x[0]);
        boost::hash_combine(seed, x[1]);
        boost::hash_combine(seed, x[2]);
        return seed;
    }
};

template<typename PT>
class VoxelMap {
public:
    using voxelMap = std::unordered_map<Eigen::Vector3i, typename pcl::PointCloud<PT>::Ptr, Vector3iHash, std::equal_to<Eigen::Vector3i>,
            Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, typename pcl::PointCloud<PT>::Ptr>>>;
    voxelMap voxels_;
    std::vector<Eigen::Vector3i> vecCoord_;
    double voxel_resolution_;

    explicit VoxelMap() {}

    void set_resolution(const double res) { voxel_resolution_ = res; }

    Eigen::Vector3i voxel_coord(const Eigen::Vector4d& x) const {
        return (x.array() / voxel_resolution_ - 0.5).floor().template cast<int>().template head<3>();
//        return (x.array() > 0).select(x / voxel_resolution_, x.array() / voxel_resolution_ - 1).template cast<int>().template head<3>();
    }

    void create_voxelmap(const pcl::PointCloud<PT>::Ptr& cloud, const Eigen::Isometry3d& Trans = Eigen::Isometry3d::Identity()) {
        // a fast clear() for voxels_
        voxelMap tmp;
        voxels_.swap(tmp);
        vecCoord_.clear();
//            vecCoord.clear();

        for(int i = 0; i < cloud->size(); i++) {
            Eigen::Vector4d ptG = cloud->at(i).getVector4fMap().template cast<double>();
            ptG = Trans * ptG;
            Eigen::Vector3i coord = voxel_coord(ptG);

            auto found = voxels_.find(coord);
            if(found == voxels_.end()) {
                typename pcl::PointCloud<PT>::Ptr voxel(new typename pcl::PointCloud<PT>() );
                found = voxels_.insert(found, std::make_pair(coord, voxel));
                vecCoord_.push_back(coord);
            }

            auto& cloudInMap = found->second;
            PT pt2Insert = cloud->at(i);
            pt2Insert.x = ptG.x();
            pt2Insert.y = ptG.y();
            pt2Insert.z = ptG.z();
            cloudInMap->push_back(pt2Insert);
        }
    }

    void map_filter(int min_num){
        for (auto iter = voxels_.begin(); iter != voxels_.end();) {
            auto voxelCloud = iter->second;
            if (voxelCloud->size() < min_num) iter = voxels_.erase(iter);
            else                        ++iter;
        }
    }
};


#endif //FAST_LIO_COMMON_HPP
