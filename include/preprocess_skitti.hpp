// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#define PCL_NO_PRECOMPILE // !! BEFORE ANY PCL INCLUDE!!


#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include <csignal>
#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <random>
#include <unordered_map>
#include <boost/functional/hash.hpp>

#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <omp.h>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>

#include <unistd.h>
#include <Python.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>

#include <yaml-cpp/yaml.h>
#include "common.hpp"



class KittiLoader{

public:
    KittiLoader() {
        myCloud_.reset(new pcl::PointCloud<PointXYZLIS>());
        cloudMulti_.reset(new pcl::PointCloud<PointXYZLIS>());
    };

    void loadParam(YAML::Node& yamlNode){
        auto kittiPath = yamlNode["semantic_kitti"]["kittiPath"].as<std::string>();
        auto vecTr = yamlNode["semantic_kitti"]["tr"].as<std::vector<double>>();
        voxRes_ = yamlNode["semantic_kitti"]["voxRes"].as<double>();
        auto cMap = yamlNode["semantic_kitti"]["color_map"];
        tr_ = Eigen::Map<Eigen::Matrix4d>(vecTr.data());
        tr_.transposeInPlace();

        auto pathPoint = kittiPath + "/velodyne";
        auto pathLabel = kittiPath + "/labels";
        auto pathStamp = kittiPath + "/times.txt";
        auto pathPoses = kittiPath + "/poses.txt";

        loadPointPath(pathPoint);
        loadLabelPath(pathLabel);
        loadStamp(pathStamp);
        loadPoses(pathPoses);
        loadColorMap(cMap);
    }

    void loadSequence(int sequence){
        myCloud_->clear();

        auto binSeqPath = vecBinFile_.at(sequence);
        auto labelSeqPath = vecLabelFile_.at(sequence);

        std::ifstream binSeqFile(binSeqPath, std::ios::binary);
        std::ifstream labelSeqFile(labelSeqPath, std::ios::binary);

        // Get the file size
        labelSeqFile.seekg(0, std::ios::end);
        std::streampos sizeLabelFile = labelSeqFile.tellg();
        labelSeqFile.seekg(0, std::ios::beg);

        // Calculate the number of vecLabel (assuming each element is 4 bytes)
        std::size_t numLabel = sizeLabelFile / 4;

        // Read the binary data into a vector of unsigned integers
        std::vector<std::uint32_t> vecLabel(numLabel);
        labelSeqFile.read(reinterpret_cast<char*>(vecLabel.data()), sizeLabelFile);

        labelSeqFile.close();

        binSeqFile.seekg(0, std::ios::end);
        std::streampos fileSizeBin = binSeqFile.tellg();
        binSeqFile.seekg(0, std::ios::beg);

        // Calculate the number of vecLabel (assuming each element is 4 bytes)
        std::size_t numBin = fileSizeBin / 4;
        std::vector<float> pointElements(numBin);
        binSeqFile.read(reinterpret_cast<char*>(pointElements.data()), fileSizeBin);

        PointXYZLIS pointTmp;

        int i = 0, f = 0;
        for (auto pointElement : pointElements) {
            if (f ==3) {
                pointTmp.intensity = pointElement;
                auto comp = vecLabel.at(i);
                pointTmp.label = static_cast<std::uint16_t>(comp & 0xFFFF);
//                pointTmp.seq    = static_cast<std::uint16_t>(comp >> 16); // the instance id
                pointTmp.seq = sequence;
                myCloud_->push_back(pointTmp);
                f = 0;
                i += 1;
                continue;
            }
            pointTmp.data[f] = pointElement;
            f += 1;
        }
        poseCurr_ = vecPoses_.at(sequence);

        auto iter = colorMap_.begin();
        std::unordered_map<uint16_t, decltype(iter)> idMap;

    }

    void concentrateSeqs(const int& seq, const int& deltaSeq, bool visualize){
        cloudMulti_.reset(new pcl::PointCloud<PointXYZLIS>());
        Eigen::Matrix4d poseOri = vecPoses_.at(seq);

        for (int i = seq; i < seq + deltaSeq; ++i) {
            loadSequence(i);

            pcl::VoxelGrid<PointXYZLIS> sor;
            sor.setInputCloud(myCloud_); // cloud is your input point cloud
            sor.setLeafSize(voxRes_, voxRes_, voxRes_); // Set the voxel grid size
            sor.filter(*myCloud_); // cloud_filtered is the resultant downsampled point cloud

            randomDownSample(myCloud_);
            for (auto & j : *myCloud_) {
                if (j.label == 80){
                    PointXYZLIS ptTmp;
                    ptTmp.x = j.x;
                    ptTmp.y = j.y;
                    ptTmp.z = j.z;
                    ptTmp.label = j.label;

                    Eigen::Vector4d ptV(ptTmp.x, ptTmp.y, ptTmp.z, 1);
                    ptV = poseOri.inverse() * poseCurr_ * ptV;
                    ptTmp.x = ptV.x();
                    ptTmp.y = ptV.y();
                    ptTmp.z = ptV.z();

                    cloudMulti_->push_back(ptTmp);
                }
            }
        }

        if (visualize)  {
            pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud2vis(new pcl::PointCloud<pcl::PointXYZRGBL>);

            for (int i = 0; i < cloudMulti_->size(); ++i) {
                pcl::PointXYZRGBL ptTmp;
                auto vecRgb = colorMap_.at(ptTmp.label);
                ptTmp.b = vecRgb.at(0) + (i-100)*20;
                ptTmp.g = vecRgb.at(1) + (i-100)*20;
                ptTmp.r = vecRgb.at(2);
                cloud2vis->push_back(ptTmp);
            }

            pcl::visualization::PCLVisualizer viewer("RGB Point Cloud");
            viewer.setBackgroundColor(0, 0, 0);
            viewer.addPointCloud<pcl::PointXYZRGBL>(cloud2vis, "cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

            while (!viewer.wasStopped()) {
                viewer.spinOnce(100);
            }

        }
    }

    // problem left: cloudIn turns void after downSample.
    void randomDownSample(pcl::PointCloud<PointXYZLIS>::Ptr& cloudIn){
        VoxelMap voxelMap;
        std::vector<Eigen::Vector3i> vecCoord;
        for (auto pt : *cloudIn) {
            Eigen::Matrix<double, 5, 1> ptVec;
            ptVec.x() = pt.x;
            ptVec.y() = pt.y;
            ptVec.z() = pt.z;
            ptVec.w() = pt.intensity;
            ptVec[4] = pt.label;
            Eigen::Vector3i coord = voxel_coord(ptVec);
            if (!voxelMap.contains(coord)){
                vecCoord.push_back(coord);
            }
            voxelMap[coord].push_back(ptVec);
        }

        std::random_device rd; // Seed generator
        std::mt19937 gen(rd()); // Mersenne Twister engine
        // Define a random number distribution (e.g., uniform distribution between 1 and 99)
        std::uniform_int_distribution<int> distribution(1, 99);

        for (auto& coord : vecCoord){
            auto vecPt = voxelMap[coord];
            int random_number = distribution(gen); // Generate a random integer
            int randomCnt = static_cast<int>((random_number / 100.0) *
                    static_cast<float>(vecPt.size()));
            voxelMap[coord].clear();
            voxelMap[coord].push_back(vecPt.at(randomCnt));
            vecPt.clear();
        }

        cloudIn->clear();
        for (auto& coord : vecCoord) {
            PointXYZLIS pt;
            auto element = voxelMap[coord];
            pt.x = element.at(0).x();
            pt.y = element.at(0).y();
            pt.z = element.at(0).z();
            pt.intensity = element.at(0).w();
            pt.label = element.at(0)[4];
            cloudIn->push_back(pt);
        }
// region visualize
//        pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud2vis(new pcl::PointCloud<pcl::PointXYZRGBL>());
//        for (auto ptIn : *cloudIn) {
//                pcl::PointXYZRGBL pt2vis;
//                pt2vis.x = ptIn.x;
//                pt2vis.y = ptIn.y;
//                pt2vis.z = ptIn.z;
//                pt2vis.b = 0;
//                pt2vis.g = 0;
//                pt2vis.r = 125;
//                cloud2vis->push_back(pt2vis);
//        }
//
//        pcl::visualization::PCLVisualizer viewer("RGB Point Cloud");
//        viewer.setBackgroundColor(0, 0, 0);
//        viewer.addPointCloud<pcl::PointXYZRGBL>(cloud2vis, "cloud");
//        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud");
//
//        while (!viewer.wasStopped()) {
//            viewer.spinOnce(100);
//        }
// endregion

    }



    Eigen::Vector3i voxel_coord(const Eigen::VectorXd& x) const {
        return (x.array().block<3, 1>(0, 0) / voxRes_ - 0.5).floor().template cast<int>().template head<3>();
    }

private:
    void loadPointPath(std::string& binPathString){
        vecBinFile_.clear();
        std::filesystem::path binPath(binPathString);
        if (!is_directory(binPath)) {
            std::cerr << "Failed to open binPath" << std::endl;
            return;
        }
        for (const auto& binFileIter : std::filesystem::directory_iterator(binPath)) {
            vecBinFile_.push_back(binFileIter.path());
        }
        std::sort(vecBinFile_.begin(), vecBinFile_.end());

    }

    void loadLabelPath(std::string& labelPathString){
        vecLabelFile_.clear();
        std::filesystem::path labelPath(labelPathString);
        if (!is_directory(labelPath)) {
            std::cerr << "Failed to open labelPath" << std::endl;
            return;
        }
        for (const auto& labelFileIter : std::filesystem::directory_iterator(labelPath)) {
            vecLabelFile_.push_back(labelFileIter.path());
        }
        std::sort(vecLabelFile_.begin(), vecLabelFile_.end());
    }

    void loadStamp(std::string& stampFileString){
        std::ifstream input_file(stampFileString);
        if (!input_file.is_open()) {
            std::cerr << "Failed to open the stampFile." << std::endl;
            return;
        }

        std::string line;
        while (std::getline(input_file, line)) {
            vecStamp_.push_back(line);
        }
    }

    void loadPoses(std::string& posesFileString){
        std::ifstream input_file(posesFileString);
        if (!input_file.is_open()) {
            std::cerr << "Failed to open the posesFile." << std::endl;
            return;
        }

        std::string line, num;
        std::vector<double> vecNums;
        while (std::getline(input_file, line)) {
            std::istringstream iss(line);
            while (std::getline(iss, num, ' ')) {
                vecNums.push_back(std::stod(num));
            }
            vecNums.push_back(0);
            vecNums.push_back(0);
            vecNums.push_back(0);
            vecNums.push_back(1);
            Eigen::Matrix4d poseTmp(vecNums.data());
            vecNums.clear();
            poseTmp.transposeInPlace();
            vecPoses_.emplace_back(tr_.inverse() * poseTmp * tr_);
        }
    }

    void loadColorMap(YAML::Node& cMap){
        for (const auto& entry : cMap) {
            auto label = entry.first.as<uint16_t>();
            auto color = entry.second.as<std::vector<uint16_t>>();
            colorMap_.insert(std::make_pair(label, color));
        }
    }

public:
    void filterLabel(const pcl::PointCloud<PointXYZLIS>::Ptr& cloudOut, uint16_t label){
        for (int i = 0; i < myCloud_->size(); ++i) {
            auto ptTmp = myCloud_->points[i];
            if (ptTmp.label == label)
                cloudOut->push_back(ptTmp);
        }
    }

public:
    std::vector<std::string> vecBinFile_, vecLabelFile_, vecStamp_;
    std::vector<Eigen::Matrix4d> vecPoses_;

    Eigen::Matrix4d tr_, poseCurr_;

    pcl::PointCloud<PointXYZLIS>::Ptr myCloud_, cloudMulti_;

    std::unordered_map<uint16_t, std::vector<uint16_t>> colorMap_;

    double voxRes_;

    using VoxelMap = std::unordered_map<Eigen::Vector3i, std::vector<Eigen::Matrix<double, 5, 1>>, Vector3iHash, std::equal_to<Eigen::Vector3i>, Eigen::aligned_allocator<std::pair<const Eigen::Vector3i, std::vector<Eigen::Matrix<double, 5, 1>>>>>;
};

void visualizeMultiFrame(KittiLoader& kittiLoader, const int& seq, const int& deltaSeq){
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr cloud2vis(new pcl::PointCloud<pcl::PointXYZRGBL>);
    cloud2vis->resize(kittiLoader.myCloud_->size());
    Eigen::Matrix4d poseOri = kittiLoader.vecPoses_.at(seq);
    for (int i = seq; i < seq + deltaSeq; ++i) {
        kittiLoader.loadSequence(i);
        for (auto & j : *kittiLoader.myCloud_) {
            if (j.label == 80){
                pcl::PointXYZRGBL ptTmp;
                ptTmp.x = j.x;
                ptTmp.y = j.y;
                ptTmp.z = j.z;
                ptTmp.label = j.label;

                Eigen::Vector4d ptV(ptTmp.x, ptTmp.y, ptTmp.z, 1);
                ptV = poseOri.inverse() * kittiLoader.poseCurr_ * ptV;
                ptTmp.x = ptV.x();
                ptTmp.y = ptV.y();
                ptTmp.z = ptV.z();
                auto vecRgb = kittiLoader.colorMap_.at(ptTmp.label);
                ptTmp.b = vecRgb.at(0) + (i-100)*20;
                ptTmp.g = vecRgb.at(1) + (i-100)*20;
                ptTmp.r = vecRgb.at(2);

                cloud2vis->push_back(ptTmp);
            }
        }
    }


    pcl::visualization::PCLVisualizer viewer("RGB Point Cloud");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addPointCloud<pcl::PointXYZRGBL>(cloud2vis, "cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

//int main(int argc, char** argv) {
//    YAML::Node config = YAML::LoadFile("/home/xingyu/catkin_ws/src/semantic_ba/config/read_label.yaml");
//    auto cMap = config["color_map"];
//
//    KittiLoader kittiLoader;
//    kittiLoader.loadParam(config);
//    kittiLoader.loadSequence(100);
//
//    visualizeMultiFrame(kittiLoader, 100, 3);
//}
