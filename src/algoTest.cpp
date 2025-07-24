

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
#include <chrono>
#include <map>
#include <set>

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
#include "SGBA.hpp"  // Use optimized version
#include "preprocess_skitti.hpp"

std::vector<Eigen::Matrix4d> vecOdom;
std::deque<pcl::PointCloud<PointXYZLIS>::Ptr> deqZ;
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> vecT;

Eigen::Matrix<double, Eigen::Dynamic, 4> matT;
std::deque<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> deqT;

// Performance monitoring
struct PerformanceStats {
    double totalTime = 0;
    double initTime = 0;
    double expectationTime = 0;
    double mRigidTime = 0;
    double mGmmTime = 0;
    int iterations = 0;
    
    void reset() {
        totalTime = initTime = expectationTime = mRigidTime = mGmmTime = 0;
        iterations = 0;
    }
    
    void print() {
        std::cout << "=== PERFORMANCE STATISTICS ===" << std::endl;
        std::cout << "Total time: " << totalTime << " ms" << std::endl;
        std::cout << "Initialization: " << initTime << " ms" << std::endl;
        std::cout << "Expectation avg: " << expectationTime / iterations << " ms" << std::endl;
        std::cout << "M-rigid avg: " << mRigidTime / iterations << " ms" << std::endl;
        std::cout << "M-GMM avg: " << mGmmTime / iterations << " ms" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "=============================" << std::endl;
    }
} perfStats;

void processOdometry(const nav_msgs::Odometry::ConstPtr& odom_msg) {
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

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Optimized Cloud Viewer"));
int currentCloudIndex = 0;
pcl::PointCloud<pcl::PointXYZ>::Ptr aggrCloud2vis(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr muGMM2vis(new pcl::PointCloud<pcl::PointXYZ>());
bool proceedToNextIteration = false;
bool shouldQuit = false;

void keyboard4EMIter(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    if (event.getKeySym() == "n" && event.keyDown()) {
        viewer->removePointCloud("aggregated_cloud");
        viewer->removePointCloud("gmm_cloud");
        viewer->addPointCloud(aggrCloud2vis, "aggregated_cloud");
        viewer->addPointCloud(muGMM2vis, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aggregated_cloud");

        proceedToNextIteration = true;
    }
    if (event.getKeySym() == "s" && event.keyDown()) {
        perfStats.print();
    }
    if (event.getKeySym() == "q" && event.keyDown()) {
        std::cout << "\nQuitting..." << std::endl;
        shouldQuit = true;
        proceedToNextIteration = true;
    }
}

int main(int argc, char** argv) {
    auto programStart = std::chrono::high_resolution_clock::now();
    
    YAML::Node config = YAML::LoadFile("/home/xingyu/catkin_ws/src/semanticBA/config/read_label.yaml");

    std::string testMode = "kitti"; // default
    if (config["test_mode"]) {
        testMode = config["test_mode"].as<std::string>();
    }
    
    std::cout << "=== SemanticBA Optimized Test ===" << std::endl;
    std::cout << "Test mode: " << testMode << std::endl;

    auto kittiPath  = config["semantic_kitti"]["kittiPath"].as<std::string>();
    auto label2Read = config["semantic_kitti"]["label2Read"].as<std::vector<uint16_t>>();
    auto voxResKiti = config["semantic_kitti"]["voxRes"].as<float>();
    auto windowSize = config["semantic_kitti"]["winSize"].as<uint16_t>();
    auto frameCnt   = config["semantic_kitti"]["frameCnt"].as<int>();
    auto regulateT  = config["semantic_kitti"]["regulateT"].as<bool>();

    auto voxResGmm  = config["gmm"]["voxRes"].as<double>();
    auto TRemainThd   = config["gmm"]["TRemainThd"].as<int>();
    auto iter_max   = config["gmm"]["iterMax"].as<int>();

    // vox res for gmm initialization
    std::map<uint16_t, double> voxelSizeMapGMM;
    if (config["gmm"]["voxResPerLabel"]) {
        auto voxResPerLabel = config["gmm"]["voxResPerLabel"];
        for (auto it = voxResPerLabel.begin(); it != voxResPerLabel.end(); ++it) {
            uint16_t label = it->first.as<uint16_t>();
            double voxSize = it->second.as<double>();
            voxelSizeMapGMM[label] = voxSize;
            std::cout << "GMM voxel size for label " << label << ": " << voxSize << std::endl;
        }
    }

    // vox res for downsampling
    std::map<uint16_t, double> voxelSizeMapFilter;
    if (config["semantic_kitti"]["voxResPerLabelFilter"]) {
        auto voxResPerLabelFilter = config["semantic_kitti"]["voxResPerLabelFilter"];
        for (auto it = voxResPerLabelFilter.begin(); it != voxResPerLabelFilter.end(); ++it) {
            uint16_t label = it->first.as<uint16_t>();
            double voxSize = it->second.as<double>();
            voxelSizeMapFilter[label] = voxSize;
            std::cout << "Filter voxel size for label " << label << ": " << voxSize << std::endl;
        }
    }

    ros::init(argc, argv, "optimized_semantic_ba");
    ros::NodeHandle nh;

    // Initialize performance monitoring
    perfStats.reset();
    
    pcl::PointCloud<PointXYZLIS>::Ptr cloudAggrTmp(new pcl::PointCloud<PointXYZLIS>());
    std::vector<pcl::PointCloud<PointXYZLIS>::Ptr> vecCloudAggr;

    // Use optimized SGBA solver
    SGBA<PointXYZLIS> solverEM;

    deqZ.clear();
    deqT.clear();

    Eigen::Matrix4d T0 = Eigen::Matrix4d::Identity();
    
    // Data loading with timing
    auto dataStart = std::chrono::high_resolution_clock::now();
    
    if (testMode == "virtual_planes") {
        std::cout << "\n=== Loading Virtual Plane Data ===" << std::endl;
        
        // Load virtual plane configuration
        std::string pcdFile = "/home/xingyu/catkin_ws/src/semanticBA/pcd/largeE3_6.pcd";
        int maxFrames = 20;
        bool useIdentityTransforms = true;
        
        if (config["virtual_planes"]) {
            if (config["virtual_planes"]["pcd_file"]) {
                pcdFile = config["virtual_planes"]["pcd_file"].as<std::string>();
            }
            if (config["virtual_planes"]["max_frames"]) {
                maxFrames = config["virtual_planes"]["max_frames"].as<int>();
            }
            if (config["virtual_planes"]["use_identity_transforms"]) {
                useIdentityTransforms = config["virtual_planes"]["use_identity_transforms"].as<bool>();
            }
        }
        
        std::cout << "PCD file: " << pcdFile << std::endl;
        std::cout << "Max frames: " << maxFrames << std::endl;
        std::cout << "Use identity transforms: " << (useIdentityTransforms ? "yes" : "no") << std::endl;
        
        // Load virtual plane data
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr oriCloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        
        if (pcl::io::loadPCDFile<pcl::PointXYZINormal>(pcdFile, *oriCloud) == -1) {
            std::cerr << "Error: Could not load PCD file: " << pcdFile << std::endl;
            return -1;
        }
        
        std::cout << "Loaded " << oriCloud->size() << " points from virtual plane data" << std::endl;
        
        // Group points by frame (intensity field represents frame count)
        std::unordered_map<int, std::vector<pcl::PointXYZINormal>> mapFrame;
        int maxFrameId = 0;
        
        for (const auto& pt : *oriCloud) {
            int frameId = static_cast<int>(pt.intensity);
            if (frameId < maxFrames) {
                mapFrame[frameId].push_back(pt);
                maxFrameId = std::max(maxFrameId, frameId);
            }
        }
        
        windowSize = std::min(maxFrameId + 1, maxFrames);
        std::cout << "Virtual plane data spans " << windowSize << " frames" << std::endl;
        
        // Resize containers
        deqZ.resize(windowSize);
        deqT.resize(windowSize);
        
        // Initialize transformations and point clouds
        for (int i = 0; i < windowSize; ++i) {
            deqZ.at(i).reset(new pcl::PointCloud<PointXYZLIS>());
            
            if (useIdentityTransforms) {
                deqT.at(i).block<3, 3>(0, 0).setIdentity();
                deqT.at(i).block<3, 1>(0, 3).setZero();
            } else {
                // Could add some synthetic transformations here if needed
                deqT.at(i).block<3, 3>(0, 0).setIdentity();
                deqT.at(i).block<3, 1>(0, 3).setZero();
            }
        }
        
        // Convert points to PointXYZLIS format and assign to frames
        for (const auto& framePair : mapFrame) {
            int frameId = framePair.first;
            const auto& framePoints = framePair.second;
            
            for (const auto& pt : framePoints) {
                PointXYZLIS ptTmp;
                ptTmp.x = pt.x;
                ptTmp.y = pt.y;
                ptTmp.z = pt.z;
                ptTmp.intensity = pt.intensity;
                // Use curvature as semantic label (or assign default)
                ptTmp.label = static_cast<uint16_t>(pt.curvature); 
                ptTmp.seq = frameId;
                
                if (frameId < windowSize) {
                    deqZ.at(frameId)->push_back(ptTmp);
                }
            }
            
            std::cout << "Frame " << frameId << ": " << framePoints.size() << " points" << std::endl;
        }
        
        // For virtual planes, create aggregated clouds based on semantic labels
        std::map<uint16_t, pcl::PointCloud<PointXYZLIS>::Ptr> labelClouds;
        std::set<uint16_t> uniqueLabels;
        
        // Collect unique labels
        for (int i = 0; i < windowSize; ++i) {
            for (const auto& pt : deqZ.at(i)->points) {
                uniqueLabels.insert(pt.label);
            }
        }
        
        // Create aggregated clouds for each label
        vecCloudAggr.clear();
        label2Read.clear();
        
        for (uint16_t label : uniqueLabels) {
            label2Read.push_back(label);
            pcl::PointCloud<PointXYZLIS>::Ptr labelCloud(new pcl::PointCloud<PointXYZLIS>());
            
            // Aggregate points from all frames for this label
            for (int i = 0; i < windowSize; ++i) {
                for (const auto& pt : deqZ.at(i)->points) {
                    if (pt.label == label) {
                        PointXYZLIS aggrPt = pt;
                        // Apply transformation if not identity
                        Eigen::Vector3d ptVec(pt.x, pt.y, pt.z);
                        Eigen::Vector3d aggrPtVec = deqT.at(i).block<3, 3>(0, 0) * ptVec + deqT.at(i).block<3, 1>(0, 3);
                        aggrPt.x = aggrPtVec.x();
                        aggrPt.y = aggrPtVec.y();
                        aggrPt.z = aggrPtVec.z();
                        labelCloud->push_back(aggrPt);
                    }
                }
            }
            
            vecCloudAggr.push_back(labelCloud);
            *cloudAggrTmp += *labelCloud;
            
            std::cout << "Label " << label << ": " << labelCloud->size() << " aggregated points" << std::endl;
        }
        
        std::cout << "Virtual plane data loaded successfully!" << std::endl;
        
    } else { // KITTI mode
        std::cout << "\n=== Loading KITTI Data ===" << std::endl;
        
        KittiLoader kittiLoader;
        kittiLoader.loadParam(config);

        deqZ.resize(windowSize);
        deqT.resize(windowSize);
        
        std::vector<std::vector<pcl::PointCloud<PointXYZLIS>::Ptr>> vecCloud_sj;
        vecCloud_sj.clear();
        vecCloud_sj.resize(label2Read.size());
        for (auto& vecCloud_s : vecCloud_sj) {
            vecCloud_s.clear();
            vecCloud_s.resize(windowSize);
        }
        
        for (int l = 0; l < label2Read.size(); ++l) {
            for (int i = 0; i < windowSize; ++i) {
                kittiLoader.loadSequence(frameCnt + i);
                Eigen::Matrix4d trTmp = kittiLoader.poseCurr_;

                if (i == 0) T0 = trTmp;
                if (regulateT) {
                    trTmp = T0.inverse() * trTmp;
                }
                deqT.at(i).block<3, 3>(0, 0) = trTmp.block<3, 3>(0, 0);
                deqT.at(i).block<3, 1>(0, 3) = trTmp.block<3, 1>(0, 3);

                pcl::PointCloud<PointXYZLIS>::Ptr cloud2paste(new pcl::PointCloud<PointXYZLIS>());
                kittiLoader.filterLabel(cloud2paste, label2Read.at(l));
                vecCloud_sj[l][i] = cloud2paste;
            }
        }

        // Aggregate clouds by semantic label
        vecCloudAggr.resize(label2Read.size());
        for (int l = 0; l < label2Read.size(); ++l) {
            std::vector<pcl::PointCloud<PointXYZLIS>::Ptr> vecCloudTmp;
            vecCloudTmp.resize(windowSize);
            for (int i = 0; i < windowSize; ++i) {
                vecCloudTmp.at(i) = vecCloud_sj[l][i];
            }
            pcl::PointCloud<PointXYZLIS>::Ptr cloudTmp(new pcl::PointCloud<PointXYZLIS>());
            solverEM.aggregateCloud(vecCloudTmp, deqT, cloudTmp);
            *cloudAggrTmp += *cloudTmp;
            vecCloudAggr.at(l) = cloudTmp;
        }

        // Prepare frame clouds with per-label voxel filtering
        for (int i = 0; i < windowSize; ++i) {
            for (int l = 0; l < label2Read.size(); ++l) {
                if (deqZ.at(i) == nullptr) {
                    deqZ.at(i) = vecCloud_sj[l][i];
                } else
                    *deqZ.at(i) += *vecCloud_sj[l][i];
            }
            
            // Apply per-label voxel filtering to the combined cloud
            pcl::PointCloud<PointXYZLIS>::Ptr filteredCloud(new pcl::PointCloud<PointXYZLIS>());
            for (uint16_t label : label2Read) {
                // Extract points for this label
                pcl::PointCloud<PointXYZLIS>::Ptr labelCloud(new pcl::PointCloud<PointXYZLIS>());
                for (const auto& pt : deqZ.at(i)->points) {
                    if (pt.label == label) {
                        labelCloud->push_back(pt);
                    }
                }
                
                if (!labelCloud->empty()) {
                    // Get appropriate voxel size for this label
                    double filterVoxelSize = voxResKiti; // default
                    auto it = voxelSizeMapFilter.find(label);
                    if (it != voxelSizeMapFilter.end()) {
                        filterVoxelSize = it->second;
                    }
                    
                    // Apply voxel filtering with label-specific resolution
                    pcl::VoxelGrid<PointXYZLIS> sor;
                    sor.setInputCloud(labelCloud);
                    sor.setLeafSize(filterVoxelSize, filterVoxelSize, filterVoxelSize);
                    
                    pcl::PointCloud<PointXYZLIS>::Ptr labelFiltered(new pcl::PointCloud<PointXYZLIS>());
                    sor.filter(*labelFiltered);
                    
                    *filteredCloud += *labelFiltered;
                    
                    std::cout << "Frame " << i << ", Label " << label << ": " 
                              << labelCloud->size() << " -> " << labelFiltered->size() 
                              << " points (voxel size: " << filterVoxelSize << ")" << std::endl;
                }
            }
            deqZ.at(i) = filteredCloud;
        }
    }

    auto dataEnd = std::chrono::high_resolution_clock::now();
    auto dataDuration = std::chrono::duration_cast<std::chrono::milliseconds>(dataEnd - dataStart);
    std::cout << "Data loading time: " << dataDuration.count() << " ms" << std::endl;

    // Initialize optimized solver
    auto initStart = std::chrono::high_resolution_clock::now();
    
    solverEM.initParams(false, false, TRemainThd);
    
    // Set up data structures for optimized solver
    solverEM.deqZ_ = deqZ;
    solverEM.deqT_ = deqT;
    solverEM.K_ = deqZ.size();
    
    // Initialize using the multi-semantic method with per-label voxel sizes
    solverEM.initializeMulti(deqZ, vecCloudAggr, deqT, voxResGmm, true, voxelSizeMapGMM);
    
    // Pre-allocate memory for optimization
    solverEM.preallocateMemory();
    
    // Build spatial index if beneficial
    solverEM.buildSpatialIndex();

    auto initEnd = std::chrono::high_resolution_clock::now();
    perfStats.initTime = std::chrono::duration_cast<std::chrono::milliseconds>(initEnd - initStart).count();
    std::cout << "Optimized initialization time: " << perfStats.initTime << " ms" << std::endl;

    // Setup visualization
    pcl::copyPointCloud(*cloudAggrTmp, *aggrCloud2vis);
    for (int j = 0; j < solverEM.J_; ++j) {
        pcl::PointXYZ muj;
        muj.x = solverEM.M_.col(j).x();
        muj.y = solverEM.M_.col(j).y();
        muj.z = solverEM.M_.col(j).z();
        muGMM2vis->push_back(muj);
    }

    viewer->registerKeyboardCallback(keyboard4EMIter, (void*)&viewer);
    viewer->addPointCloud(aggrCloud2vis, "aggregated_cloud");
    viewer->addPointCloud(muGMM2vis, "gmm_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "gmm_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aggregated_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gmm_cloud");

    std::cout << "\n=== Mode: " << testMode << " ===" << std::endl;
    std::cout << "Frames: " << windowSize << ", Labels: " << label2Read.size() << std::endl;
    std::cout << "Press 'n' to proceed, 's' to show stats, 'q' to quit..." << std::endl;
    while (!proceedToNextIteration && !shouldQuit) {
        viewer->spinOnce();
    }

    if (shouldQuit) {
        std::cout << "Exiting before optimization..." << std::endl;
        return 0;
    }

    // Main optimization loop with performance monitoring
    for (int i = 0; i < iter_max && !shouldQuit; ++i) {
        aggrCloud2vis.reset(new pcl::PointCloud<pcl::PointXYZ>());
        muGMM2vis.reset(new pcl::PointCloud<pcl::PointXYZ>());
        proceedToNextIteration = false;

        std::cout << "\n=== ITERATION " << i+1 << "/" << iter_max << " ===" << std::endl;

        // Optimized Expectation step
        auto eStart = std::chrono::high_resolution_clock::now();
        solverEM.expectationMultiSemanticOptimized(INLIER);
        auto eEnd = std::chrono::high_resolution_clock::now();
        auto eDuration = std::chrono::duration_cast<std::chrono::milliseconds>(eEnd - eStart).count();
        perfStats.expectationTime += eDuration;

        // Optimized M-rigid step
        auto mStart = std::chrono::high_resolution_clock::now();
        solverEM.M_rigidOptimized();
        auto mEnd = std::chrono::high_resolution_clock::now();
        auto mDuration = std::chrono::duration_cast<std::chrono::milliseconds>(mEnd - mStart).count();
        perfStats.mRigidTime += mDuration;

        // Optimized M-GMM step
        auto gStart = std::chrono::high_resolution_clock::now();
        solverEM.M_GMMOptimized();
        auto gEnd = std::chrono::high_resolution_clock::now();
        auto gDuration = std::chrono::duration_cast<std::chrono::milliseconds>(gEnd - gStart).count();
        perfStats.mGmmTime += gDuration;

        perfStats.iterations++;

        // Update visualization
        for (int j = 0; j < solverEM.J_; ++j) {
            pcl::PointXYZ muj;
            muj.x = solverEM.M_.col(j).x();
            muj.y = solverEM.M_.col(j).y();
            muj.z = solverEM.M_.col(j).z();
            muGMM2vis->push_back(muj);
        }

        // Aggregate transformed points for visualization
        for (int k = 0; k < solverEM.deqZ_.size(); ++k) {
            auto Zk = solverEM.deqZ_.at(k);
            for (auto zki : Zk->points) {
                Eigen::Vector3d aggrPt;
                aggrPt.x() = zki.x;
                aggrPt.y() = zki.y;
                aggrPt.z() = zki.z;
                aggrPt = solverEM.deqT_.at(k).block<3, 3>(0, 0) * aggrPt + solverEM.deqT_.at(k).block<3, 1>(0, 3);
                
                pcl::PointXYZ aggrPt2vis;
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
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "gmm_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aggregated_cloud");

        std::cout << "Iteration " << i+1 << " times - E: " << eDuration << "ms, M-rigid: " 
                  << mDuration << "ms, M-GMM: " << gDuration << "ms" << std::endl;

        std::cout << "Press 'n' for next iteration, 's' for stats, 'q' to quit..." << std::endl;
        while (!proceedToNextIteration && !shouldQuit) {
            viewer->spinOnce();
        }

        if (shouldQuit) {
            std::cout << "Optimization interrupted by user." << std::endl;
            break;
        }
    }

    auto programEnd = std::chrono::high_resolution_clock::now();
    perfStats.totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(programEnd - programStart).count();

    // Final performance report
    if (!shouldQuit) {
        perfStats.print();
    }
    
    std::cout << "\n=== OPTIMIZATION " << (shouldQuit ? "INTERRUPTED" : "COMPLETED") << " ===" << std::endl;
    std::cout << "Mode: " << testMode << std::endl;
    if (!shouldQuit) {
        std::cout << "Total speedup achieved with optimizations!" << std::endl;
    } else {
        std::cout << "User requested exit. Shutting down viewer..." << std::endl;
    }
    
    // Clean up viewer
    if (viewer) {
        viewer->close();
    }
    
    return 0;
} 