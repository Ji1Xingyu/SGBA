//
// Created by xingyu on 29/9/23.
//
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
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <yaml-cpp/yaml.h>
#include "common.hpp"
#include "SGBA.hpp"
#include "preprocess_skitti.hpp"

std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> vecOdom;
// each element of the vector should contain a pcl with all the labels.
std::deque<pcl::PointCloud<PointXYZLIS>::Ptr> vecV;

// matT is not suitable for pop&push operation, thus change to dequeue, size is of windowSize
std::deque<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> deqT;
std::vector<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> vecTrMarginalized;

void processOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg) {
    // Extract relevant information from the Odometry message
    geometry_msgs::Pose pose = odom_msg->pose.pose;
    geometry_msgs::Quaternion quat = pose.orientation;

    Eigen::Quaterniond quatEigen;
    quatEigen.x() = quat.x;
    quatEigen.y() = quat.y;
    quatEigen.z() = quat.z;
    quatEigen.w() = quat.w;
    Eigen::Matrix4d Tcurr = Eigen::Matrix4d::Identity();
    Tcurr.block<3, 3>(0, 0) = quatEigen.toRotationMatrix();
    Tcurr.block<3, 1>(0, 3).x() = pose.position.x;
    Tcurr.block<3, 1>(0, 3).y() = pose.position.y;
    Tcurr.block<3, 1>(0, 3).z() = pose.position.z;

    vecOdom.emplace_back(Tcurr);
}



pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
int currentCloudIndex = 0; // Index of the currently displayed clou
pcl::PointCloud<pcl::PointXYZ>::Ptr aggrCloud2vis(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr muGMM2vis(new pcl::PointCloud<pcl::PointXYZ>());
bool proceedToNextIteration = false; // Variable to control iteration continuation


void keyboard4EMIter(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    if (event.getKeySym() == "n" && event.keyDown()) { // Press 'n' to slide the window
        viewer->removePointCloud("aggregated_cloud");
        viewer->removePointCloud("gmm_cloud");
        viewer->addPointCloud(aggrCloud2vis, "aggregated_cloud");
        viewer->addPointCloud(muGMM2vis, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "gmm_cloud");  // Red color
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gmm_cloud");  // Set point size to 2
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aggregated_cloud");  // Set point size to 2

        proceedToNextIteration = true; // Set the flag to proceed to the next iteration
    }
}

int main(int argc, char** argv) {

    YAML::Node config = YAML::LoadFile("/home/xingyu/catkin_ws/src/semanticBA/config/read_label.yaml");

    auto kittiPath  = config["semantic_kitti"]["kittiPath"].as<std::string>();
    auto label2Read = config["semantic_kitti"]["label2Read"].as<std::vector<uint16_t>>();
    auto voxResKiti = config["semantic_kitti"]["voxRes"].as<float>();
    auto windowSize = config["semantic_kitti"]["winSize"].as<uint16_t>();
    auto frameCnt   = config["semantic_kitti"]["frameCnt"].as<int>();
    auto regulateT  = config["semantic_kitti"]["regulateT"].as<bool>();
    auto disKeyThd  = config["semantic_kitti"]["disKeyThd"].as<double>();
    auto angKeyThd  = config["semantic_kitti"]["angKeyThd"].as<double>();

    auto voxResGmm  = config["gmm"]["voxRes"].as<double>();
    auto iter_max   = config["gmm"]["iterMax"].as<int>();
    auto TRemainThd   = config["gmm"]["TRemainThd"].as<int>();

    std::ofstream file("/home/xingyu/catkin_ws/output/experiments_BA/kitti_00/sba.txt");
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << "/home/xingyu/catkin_ws/output/experiments_BA/kitti_00/sba.txt" << std::endl;
        return 1;
    }

    ros::init(argc, argv, "bag_reader");
    ros::NodeHandle nh;

// region read_odom
    rosbag::Bag bag;
    std::string posePath = kittiPath + "/odom_floam.bag";
    bag.open(posePath, rosbag::bagmode::Read);

    // Create a view for the specified topics
    rosbag::View view(bag, rosbag::TopicQuery("/odom"));

    // Loop through the messages in the bag
    vecOdom.emplace_back(Eigen::Matrix4d::Identity());
    for (rosbag::MessageInstance const& msg : view) {
        // Handle Odometry messages
        if (msg.isType<nav_msgs::Odometry>()) {
            nav_msgs::Odometry::ConstPtr odom_msg = msg.instantiate<nav_msgs::Odometry>();
            if (odom_msg != nullptr) {
                // Process Odometry messages
                processOdometry(odom_msg);
            }
        }
    }
    // Close the bag
    bag.close();
// endregion

// region keyFrame_selection

    auto TLast = vecOdom.at(0);
    std::vector<int> vecIdxKeyFrame;
    std::vector<Eigen::Matrix4d> vecTKey;
    vecTKey.push_back(TLast);
    vecIdxKeyFrame.push_back(0);
    for (int i = 1; i < vecOdom.size(); ++i) {
        auto TCurr = vecOdom.at(i);
        auto TDiff = TCurr.inverse() * TLast;
        double tDiff = TDiff.block<3, 1>(0, 3).norm();
        auto angleDiff = TDiff.block<3, 3>(0, 0).eulerAngles(1, 2, 0);
        double sumAngleDiff = abs(angleDiff.x()) + abs(angleDiff.y()) + abs(angleDiff.z());

        if (tDiff > disKeyThd || sumAngleDiff > angKeyThd) {
            // regarded as keyFrame
            vecIdxKeyFrame.push_back(i);
            vecTKey.push_back(TCurr);
            TLast = TCurr;
        }
    }
    std::cout << "size of Ori seq: " << vecOdom.size() << std::endl;
    std::cout << "size of Key seq: " << vecTKey.size() << " " << vecIdxKeyFrame.size() << std::endl;

// endregion

    pcl::PointCloud<PointXYZLIS>::Ptr cloudAggrTmp(new pcl::PointCloud<PointXYZLIS>());

    std::vector<pcl::PointCloud<PointXYZLIS>::Ptr> vecCloudAggr;

    SGBA<PointXYZLIS> solverEM;

    KittiLoader kittiLoader;
    kittiLoader.loadParam(config);

    Eigen::Matrix4d T0 = Eigen::Matrix4d::Identity();

    std::deque<std::deque<pcl::PointCloud<PointXYZLIS>::Ptr>> deqCloud_sj;
    deqCloud_sj.clear();
    deqCloud_sj.resize(label2Read.size());
//    for (int l = 0; l < label2Read.size(); ++l) {
//        deqCloud_sj.at(l).resize(windowSize);
//    }
    deqT.clear();

    viewer->registerKeyboardCallback(keyboard4EMIter, (void*)&viewer);
    viewer->addPointCloud(aggrCloud2vis, "aggregated_cloud");
    viewer->addPointCloud(muGMM2vis, "gmm_cloud");

    // shift the sliding window over keyFrames: vecIdxKeyFrame, vecTKey.
    for (int keyIter = 0; keyIter < vecTKey.size(); ++keyIter) {
        cloudAggrTmp->clear();
        muGMM2vis->clear();
        aggrCloud2vis->clear();

        vecV.clear();
        vecV.resize(windowSize);
        // fill the window to the windowSize
        if (keyIter < windowSize) {
            deqT.push_back(vecTKey.at(keyIter).block<3, 4>(0, 0));
            for (int l = 0; l < label2Read.size(); ++l) {
                pcl::PointCloud<PointXYZLIS>::Ptr cloud2paste(new pcl::PointCloud<PointXYZLIS>());
                kittiLoader.loadSequence(vecIdxKeyFrame.at(keyIter));
                kittiLoader.filterLabel(cloud2paste, label2Read.at(l));
                deqCloud_sj.at(l).push_back(cloud2paste);
            }
            continue;
        }

        // vecCloudAggr
        vecCloudAggr.resize(label2Read.size());
        for (int l = 0; l < label2Read.size(); ++l) {
            std::vector<pcl::PointCloud<PointXYZLIS>::Ptr> vecCloudTmp;
            vecCloudTmp.resize(windowSize);
            for (int i = 0; i < windowSize; ++i) {
                vecCloudTmp.at(i) = deqCloud_sj[l][i];
            }
            pcl::PointCloud<PointXYZLIS>::Ptr cloudTmp(new pcl::PointCloud<PointXYZLIS>());
            solverEM.aggregateCloud(vecCloudTmp, deqT, cloudTmp);
            *cloudAggrTmp += *cloudTmp;
            vecCloudAggr.at(l) = cloudTmp;
        }

        // vecV
        for (int i = 0; i < windowSize; ++i) {
            for (int l = 0; l < label2Read.size(); ++l) {
                if (vecV.at(i) == nullptr) {
                    vecV.at(i) = deqCloud_sj[l][i];
                } else
                    *vecV.at(i) += *deqCloud_sj[l][i];
            }
            // voxel filtear the semantic cloud
            pcl::VoxelGrid<PointXYZLIS> voxFilter;
            voxFilter.setInputCloud(vecV.at(i)); // cloud is your input point cloud
            voxFilter.setLeafSize(voxResKiti, voxResKiti, voxResKiti); // Set the voxel grid size
            voxFilter.filter(*vecV.at(i)); // cloud_filtered is the resultant downsampled point cloud
        }

        auto startT = std::chrono::steady_clock::now();

        solverEM.initParams(false, false, TRemainThd);
        solverEM.initializeMulti(vecV, vecCloudAggr, deqT, voxResGmm, true);

        auto endT = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT).count();
        std::cout << "Initialization execution time: " << duration << " ms" << std::endl;

// region viewer
        pcl::copyPointCloud(*cloudAggrTmp, *aggrCloud2vis);
        for (int j = 0; j < solverEM.J_; ++j) {
            pcl::PointXYZ muj;
            muj.x = solverEM.M_.col(j).x();
            muj.y = solverEM.M_.col(j).y();
            muj.z = solverEM.M_.col(j).z();
            muGMM2vis->push_back(muj);
        }

        viewer->removePointCloud("aggregated_cloud");
        viewer->removePointCloud("gmm_cloud");
        viewer->addPointCloud(aggrCloud2vis, "aggregated_cloud");
        viewer->addPointCloud(muGMM2vis, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "gmm_cloud");  // Red color
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aggregated_cloud");  // Set point size to 2
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gmm_cloud");  // Set point size to 2
        std::cout << "Press 'n' to slide the window..." << std::endl;
        while (!proceedToNextIteration){
            viewer->spinOnce(); // Process events to update the visualization
        }

// endregion

        // BA
        std::deque<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> deqTCached;
        deqTCached.resize(windowSize);
        for (int i = 0; i < iter_max; ++i) {
            solverEM.expectationMultiSemantic(INLIER);

//            endT = std::chrono::steady_clock::now();
//            duration = std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT).count();
//            std::cout << "Expectation execution time: " << duration << " ms" << std::endl;
            solverEM.M_rigid();
//            startT = std::chrono::steady_clock::now();
//            duration = std::chrono::duration_cast<std::chrono::milliseconds>(startT - endT).count();
//            std::cout << "M_Rigid execution time: " << duration << " ms" << std::endl;
            solverEM.M_GMM();
//            endT = std::chrono::steady_clock::now();
//            duration = std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT).count();
//            std::cout << "M_GMM execution time: " << duration << " ms" << std::endl;

            if (i == 0)
                deqTCached = solverEM.deqT_;
            else {
                Eigen::Matrix4d TCurrItr = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d TCached = Eigen::Matrix4d::Identity();
                Eigen::Matrix4d TDiffItr = Eigen::Matrix4d::Identity();
                double tDiffItr, angDiffItr;
                for (int j = 0; j < windowSize; ++j) {
                    TCurrItr.block<3, 4>(0, 0) = solverEM.deqT_.at(j);
                    TCached.block<3, 4>(0, 0) = deqTCached.at(j);
                    TDiffItr = TCached * TCurrItr.inverse();

                    auto euDiff = TDiffItr.block<3, 3>(0, 0).eulerAngles(1, 2, 0);
                    angDiffItr += abs(euDiff.x()) + abs(euDiff.y()) + abs(euDiff.z());
                    tDiffItr += TDiffItr.block<3, 1>(0, 3).norm();
                    int testt = 1;
                }
                tDiffItr = tDiffItr / static_cast<double>(windowSize);
                angDiffItr = angDiffItr / static_cast<double>(windowSize);
                std::cout << "tDiff = " << tDiffItr << std::endl;
                std::cout << "angDiff = " << angDiffItr << std::endl;
                deqTCached = solverEM.deqT_;
            }
        }
// region viewer
        aggrCloud2vis.reset(new pcl::PointCloud<pcl::PointXYZ>());
        muGMM2vis.reset(new pcl::PointCloud<pcl::PointXYZ>());
        proceedToNextIteration = false;
        pcl::PointXYZ aggrPt2vis;
        Eigen::Vector3d aggrPt;
        for (int j = 0; j < solverEM.J_; ++j) {
            pcl::PointXYZ muj;
            muj.x = solverEM.M_.col(j).x();
            muj.y = solverEM.M_.col(j).y();
            muj.z = solverEM.M_.col(j).z();
            muGMM2vis->push_back(muj);
        }
        for (int k = 0; k < solverEM.deqZ_.size(); ++k) {
            auto Zk = solverEM.deqZ_.at(k);
            for (auto zki: Zk->points) {
                aggrPt.x() = zki.x;
                aggrPt.y() = zki.y;
                aggrPt.z() = zki.z;
                aggrPt = solverEM.deqT_.at(k).block<3, 3>(0, 0) * aggrPt + solverEM.deqT_.at(k).block<3, 1>(0, 3);
                aggrPt2vis.x = aggrPt.x();
                aggrPt2vis.y = aggrPt.y();
                aggrPt2vis.z = aggrPt.z();
                aggrCloud2vis->push_back(aggrPt2vis);
            }
        }
        viewer->removePointCloud("aggregated_cloud");
        viewer->removePointCloud("gmm_cloud");
        viewer->addPointCloud(aggrCloud2vis, "aggregated_cloud");
        viewer->addPointCloud(muGMM2vis, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "gmm_cloud");  // Red color
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gmm_cloud");  // Set point size to 2
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aggregated_cloud");  // Set point size to 2

        // Wait for a key press to proceed to the next iteration
        std::cout << "Press 'n' to slide the window..." << std::endl;
        while (!proceedToNextIteration){
            viewer->spinOnce(); // Process events to update the visualization
        }
// endregion


        // Marginalize

        // deqT: pop_front
        vecTrMarginalized.push_back(deqT.front());
        deqT.pop_front();
        // deqT: push_back
        deqT.push_back(vecTKey.at(keyIter).block<3, 4>(0, 0));
        std::cout << "Frame " << vecTrMarginalized.size() << " marginalized." << std::endl;


        // deqCloud_sj: pop_front & push_back
        for (int l = 0; l < label2Read.size(); ++l) {
            deqCloud_sj.at(l).pop_front();

            pcl::PointCloud<PointXYZLIS>::Ptr cloud2paste(new pcl::PointCloud<PointXYZLIS>());
            kittiLoader.loadSequence(vecIdxKeyFrame.at(keyIter));
            kittiLoader.filterLabel(cloud2paste, label2Read.at(l));
            deqCloud_sj.at(l).push_back(cloud2paste);
        }

        if (keyIter == vecTKey.size() - 1) {
            while (!deqT.empty()) {
                vecTrMarginalized.push_back(deqT.front());
                deqT.pop_front();
            }
        }
    }

    std::ofstream file_floam("/home/xingyu/catkin_ws/output/experiments_BA/kitti_00/floam.txt");
    for (size_t i = 0; i < vecOdom.size(); ++i) {
        auto mat = vecOdom[i];

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                file_floam << mat(row, col) << " ";
            }
        }
        file_floam << "\n";
    }

    file_floam.close();


    for (size_t i = 0; i < vecTrMarginalized.size(); ++i) {
        auto mat = vecTrMarginalized[i];

        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
                file << mat(row, col) << " ";
            }
        }
        file << "\n";
    }

    file.close();

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/map_after_ba", 1);
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = "base_link"; // Replace "base_link" with your desired frame ID
    cloud_msg.header.stamp = ros::Time::now(); // Use current ROS time

    pcl::PointCloud<PointXYZLIS>::Ptr cloudMap(new pcl::PointCloud<PointXYZLIS>());
    for (int i = 0; i < vecTrMarginalized.size(); ++i) {
        kittiLoader.loadSequence(i);
        auto cloud2paste = kittiLoader.myCloud_;

        *cloudMap += *cloud2paste;

        pcl::VoxelGrid<PointXYZLIS> voxFilter;
        voxFilter.setInputCloud(cloudMap); // cloud is your input point cloud
        voxFilter.setLeafSize(voxResKiti, voxResKiti, voxResKiti); // Set the voxel grid size
        voxFilter.filter(*cloudMap); // cloud_filtered is the resultant downsampled point cloud
    }

    pcl::toROSMsg(*cloudMap, cloud_msg);
    ros::Rate loop_rate(1); // Adjust the publishing rate as needed
    while (ros::ok()) {
        pub.publish(cloud_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 1;
}
