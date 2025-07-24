

#ifndef SEMANTIC_BA_SGBA_HPP
#define SEMANTIC_BA_SGBA_HPP

#include "common.hpp"
#include <omp.h>
#include <chrono>
#include <map>

enum POSTERIOR {NORMAL = 1, INLIER};

template<typename PT>
class SGBA {
public:
    SGBA() : isInitialized_(false), maxPointsPerFrame_(0) {}

    void initParams(bool tensor_yes, bool updateP, std::uint16_t TremainThd) {
        TremainThd_ = TremainThd;
        tensor_yes_ = tensor_yes;
        updateP_ = updateP;
        h_ = 2.0 / 1.0;
        gamma_ = 0.1;
        
        // Set OpenMP threads based on available cores
        int num_threads = std::min(omp_get_max_threads(), 24);
        omp_set_num_threads(num_threads);
    }


    void aggregateCloud(const std::vector<typename pcl::PointCloud<PT>::Ptr>& vecCloud,
                        const std::deque<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>& matT,
                        typename pcl::PointCloud<PT>::Ptr& cloudOut){

        cloudOut.reset(new pcl::PointCloud<PT>());

        for (int j = 0; j < vecCloud.size(); ++j) {
            auto cloudj = vecCloud.at(j);
            Eigen::Matrix3d Rj = matT.at(j).block<3, 3>(0, 0);
            Eigen::Vector3d tj = matT.at(j).block<3, 1>(0, 3);

            Eigen::Vector3d ptV, ptVG;
            PT ptG;
            for (const auto pt : cloudj->points) {
                ptG = pt;
                ptV.x() = pt.x;
                ptV.y() = pt.y;
                ptV.z() = pt.z;
                ptVG = Rj * ptV + tj;
                ptG.x = ptVG.x();
                ptG.y = ptVG.y();
                ptG.z = ptVG.z();
                cloudOut->push_back(ptG);
            }
        }
    }


    void aggregateCloud(const std::vector<typename pcl::PointCloud<PT>::Ptr>& vecCloud,
                        typename pcl::PointCloud<PT>::Ptr& cloudOut){

        cloudOut.reset(new pcl::PointCloud<PT>());

        for (int j = 0; j < vecCloud.size(); ++j) {
            auto cloudj = vecCloud.at(j);
            Eigen::Matrix3d Rj = deqT_.at(j).block<3, 3>(0, 0);
            Eigen::Vector3d tj = deqT_.at(j).block<3, 1>(0, 3);
            Eigen::Vector3d ptV, ptVG;
            PT ptG;
            for (const auto pt : cloudj->points) {
                ptV.x() = pt.x;
                ptV.y() = pt.y;
                ptV.z() = pt.z;
                ptVG = Rj * ptV + tj;
                ptG.x = ptVG.x();
                ptG.y = ptVG.y();
                ptG.z = ptVG.z();
                cloudOut->push_back(ptG);
            }
        }
    }

    /**
     *  @brief  initialize the EM solver by the voxelized, aggregated cloud.
     *          1. load the vecSemantic_ (j * 1). Thus, the cloud should be voxelized separately based on semantic labels --> vecAggr voxelized elemently.
     *
     *  @param  vecCloud - (k * Ni) vector of Z_k, including different semantic label.
     *  @param  vecAggr - (s * Ni) vector of aggregated cloud, indexed by the semantic label.
     *  @param  deqT - matrix of T_k, transformation matrices for K scans
     *  @param  voxel_size - default voxel size (fallback)
     *  @param  voxelSizeMap - per-label voxel sizes
     *
     */
    void initializeMulti (const std::deque<typename pcl::PointCloud<PT>::Ptr>& vecCloud, 
                          const std::vector<typename pcl::PointCloud<PT>::Ptr>& vecAggr,
                          const std::deque<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>>& deqT,
                          double voxel_size, bool isotropic,
                          const std::map<uint16_t, double>& voxelSizeMap = std::map<uint16_t, double>()) {
        // initialize Tk and Zk (no need to change)
        deqT_ = deqT;
        deqZ_ = vecCloud;
        iterCnt_ = 0;
        K_ = deqZ_.size();

        // Validate input
        if (K_ == 0) {
            std::cerr << "Error: No input clouds provided!" << std::endl;
            return;
        }

        J_ = 0;
        N_ = 0;
        size_t JTmp = 0, NTmp = 0, jTmp = 0;
        std::vector<Eigen::Vector3d> vecMuTmp;
        
        // Find max points per frame for matrix allocation
        maxPointsPerFrame_ = 0;
        for (const auto& cloud : deqZ_) {
            if (cloud && cloud->size() > 0) {
                maxPointsPerFrame_ = std::max(maxPointsPerFrame_, static_cast<int>(cloud->size()));
            }
        }

        if (maxPointsPerFrame_ == 0) {
            std::cerr << "Error: No valid point clouds found!" << std::endl;
            return;
        }

        std::cout << "Max points per frame: " << maxPointsPerFrame_ << std::endl;
        
        for (size_t labelIdx = 0; labelIdx < vecAggr.size(); ++labelIdx) {
            auto cloudSingleSemantic = vecAggr[labelIdx];
            if (!cloudSingleSemantic || cloudSingleSemantic->empty()) {
                std::cout << "Warning: Empty semantic cloud encountered, skipping..." << std::endl;
                continue;
            }

            // Determine the semantic label for this cloud
            uint16_t semanticLabel = 0;
            if (!cloudSingleSemantic->empty()) {
                semanticLabel = cloudSingleSemantic->points[0].label;
            }

            // Get the appropriate voxel size for this semantic label
            double currentVoxelSize = voxel_size; // default fallback
            auto voxelSizeIt = voxelSizeMap.find(semanticLabel);
            if (voxelSizeIt != voxelSizeMap.end()) {
                currentVoxelSize = voxelSizeIt->second;
                std::cout << "Using voxel size " << currentVoxelSize << " for semantic label " 
                          << semanticLabel << " (" << getLabelName(semanticLabel) << ")" << std::endl;
            } else {
                std::cout << "Using default voxel size " << currentVoxelSize << " for semantic label " 
                          << semanticLabel << " (no specific size configured)" << std::endl;
            }

            voxelMap_.set_resolution(currentVoxelSize);
            voxelMap_.create_voxelmap(cloudSingleSemantic);
            // test comment
            voxelMap_.map_filter(10);

            J_ += voxelMap_.voxels_.size();
            N_ += cloudSingleSemantic->size();

            std::cout << "Label " << semanticLabel << ": Generated " << voxelMap_.voxels_.size() 
                      << " clusters with voxel size " << currentVoxelSize << std::endl;

            auto iterMap = voxelMap_.voxels_.begin();
            /*************** IMPORTANT NOTE: CAN REMOVE THE VOXEL WITH NO OVERLAPPING POINTS FROM ALL VIEWS **************************/
            for (; jTmp < J_; ++jTmp) {
                auto cloud = iterMap->second;
                if (!cloud || cloud->empty()) {
                    ++iterMap;
                    continue;
                }

                Eigen::MatrixXd matCloud(3, cloud->size());

                for (size_t cnt = 0; cnt < cloud->size(); ++cnt) {
                    matCloud.col(cnt) = cloud->at(cnt).getVector3fMap().template cast<double>();
                }
                Eigen::Vector3d muTmp = matCloud.rowwise().mean().eval();
                matCloud.colwise() -= muTmp;
                Eigen::Matrix3d cov = matCloud * matCloud.transpose() / cloud->size();
                //to avoid singularity issues
                cov = cov + Eigen::Matrix3d::Identity() * 0.0001;
                //the EigenValue and EigenVector are sorted ascending
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> EigenSolv(cov);
                //Eigen value in INCREASING order
                Eigen::RowVector3d eigenVal = EigenSolv.eigenvalues();
                //Eigen vector corresponds to Eigen value: small----->large
                Eigen::Matrix3d eigenVec = EigenSolv.eigenvectors();

                vecMuTmp.push_back(muTmp);
                if (isotropic)  {
                    // something wrong with the Eigen decomposition, eigenVal too small.
                    // test
                    vecSigma_.push_back(1.0);
                    vecSemantic_.push_back(cloud->points[0].label);
                } else {
                    // only provide isotropic cov now
                }
                ++iterMap;
            }
        }

        // Validate cluster count
        if (J_ == 0) {
            std::cerr << "Error: No clusters generated! Check voxel resolution and data." << std::endl;
            return;
        }

        std::cout << "Initialized with " << J_ << " clusters and " << K_ << " frames" << std::endl;
        std::cout << "Cluster distribution by label:" << std::endl;
        std::map<uint16_t, int> labelCounts;
        for (const auto& label : vecSemantic_) {
            labelCounts[label]++;
        }
        for (const auto& pair : labelCounts) {
            std::cout << "  Label " << pair.first << " (" << getLabelName(pair.first) 
                      << "): " << pair.second << " clusters" << std::endl;
        }

        // initialize matrix dimensions
        P_.resize(J_);
        vecCoZ_.resize(J_);
        M_.resize(3, J_);
        M_.setZero();
        for (int j = 0; j < J_; ++j) {
            M_.col(j) = vecMuTmp.at(j);
            P_(j) = 1.0 / (J_ * (gamma_ + 1));
        }

        // Mark as initialized
        isInitialized_ = true;
        std::cout << "SGBA initialization completed successfully!" << std::endl;
    }

    void expectationMultiSemanticOptimized(const POSTERIOR& postType) {
        if (!isInitialized_) {
            std::cerr << "Error: SGBA not initialized! Call initializeMulti() first." << std::endl;
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        resetParamsInProgressOptimized();
        
        // Pre-allocate matrices instead of using hash maps (now safe after initialization)
        if (J_ > 0 && maxPointsPerFrame_ > 0) {
            try {
                alphaMatrix_.resize(K_, maxPointsPerFrame_ * J_);
                betaMatrix_.resize(K_, maxPointsPerFrame_ * J_);
                alphaMatrix_.setZero();
                betaMatrix_.setZero();
                std::cout << "Allocated matrices: " << K_ << " x " << (maxPointsPerFrame_ * J_) << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error allocating matrices: " << e.what() << std::endl;
                return;
            }
                    } else {
            std::cerr << "Error: Invalid dimensions for matrix allocation: J_=" << J_ 
                      << ", maxPointsPerFrame_=" << maxPointsPerFrame_ << std::endl;
            return;
        }
        
        // Precompute cluster constants
        precomputeClusterConstants();
        
        // Process each frame in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < K_; ++k) {
            processFrameVectorized(k);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Optimized Expectation time: " << duration.count() << " ms" << std::endl;
    }

private:
    void resetParamsInProgressOptimized() {
        ++iterCnt_;
        
        // Validate dimensions before resizing
        if (K_ == 0 || J_ == 0) {
            std::cerr << "Error: Invalid dimensions in resetParams: K_=" << K_ 
                      << ", J_=" << J_ << std::endl;
            return;
        }

        Lambda_.resize(K_, J_);
        W_.resize(3 * K_, J_);
        b_.resize(K_, J_);
        mW_.resize(3, K_);
        mX_.resize(3, K_);
        sumWeights.resize(K_);
        
        Lambda_.setZero();
        W_.setZero();
        mW_.setZero();
        mX_.setZero();
        sumWeights.setZero();
    }
    
    void precomputeClusterConstants() {
        if (J_ == 0) {
            std::cerr << "Error: Cannot precompute constants with J_=0" << std::endl;
            return;
        }

        clusterConstants_.resize(J_);
        sqrtSigmaCubed_.resize(J_);
        invTwoSigma_.resize(J_);
        
        for (int j = 0; j < J_; ++j) {
            double sigmaj = vecSigma_.at(j);
            sqrtSigmaCubed_[j] = P_(j) * sqrt(pow(sigmaj, 3));
            invTwoSigma_[j] = sigmaj / 2.0;
        }
    }
    
    void processFrameVectorized(int k) {
        auto Zk = deqZ_.at(k);
        int numPoints = Zk->size();
        
        if (numPoints == 0) return;
        
        // Get transformation for frame k
        Eigen::Matrix3d Rk = deqT_.at(k).block<3, 3>(0, 0);
        Eigen::Vector3d tk = deqT_.at(k).block<3, 1>(0, 3);
        
        // Convert point cloud to matrix for vectorized operations
        Eigen::MatrixXd pointMatrix(3, numPoints);
        std::vector<uint16_t> pointLabels(numPoints);
        
        for (int i = 0; i < numPoints; ++i) {
            pointMatrix.col(i) << Zk->points[i].x, Zk->points[i].y, Zk->points[i].z;
            pointLabels[i] = Zk->points[i].label;
        }
        
        // Transform all points at once
        Eigen::MatrixXd transformedPoints = Rk * pointMatrix;
        transformedPoints.colwise() += tk;
        
        // Compute distances to all clusters for all points
        computeDistancesVectorized(k, transformedPoints, pointLabels, numPoints);
    }
    
    void computeDistancesVectorized(int k, const Eigen::MatrixXd& transformedPoints, 
                                  const std::vector<uint16_t>& pointLabels, int numPoints) {
        
        // Pre-allocate temporary storage
        Eigen::VectorXd betaRow(J_);
        Eigen::VectorXd betaSums(numPoints);
        betaSums.setZero();
        
        // Compute beta values for all point-cluster pairs
        for (int i = 0; i < numPoints; ++i) {
            Eigen::Vector3d point = transformedPoints.col(i);
            uint16_t pointLabel = pointLabels[i];
            
            for (int j = 0; j < J_; ++j) {
                if (pointLabel != vecSemantic_.at(j)) {
                    betaRow(j) = 0.0;
                } else {
                    // Vectorized distance computation
                    double distSq = (point - M_.col(j)).squaredNorm();
                    betaRow(j) = sqrtSigmaCubed_[j] * exp(-distSq * invTwoSigma_[j]);
                }
                betaSums(i) += betaRow(j);
            }
            
            // Store beta values and compute alpha values (with bounds checking)
            int baseIdx = i * J_;
            if (baseIdx + J_ <= alphaMatrix_.cols()) {
                for (int j = 0; j < J_; ++j) {
                    betaMatrix_(k, baseIdx + j) = betaRow(j);
                    
                    // Compute alpha with outlier model
                    double alpha = betaRow(j) / (betaSums(i) + gamma_ / (h_ * (gamma_ + 1)));
                    alphaMatrix_(k, baseIdx + j) = alpha;
                }
            } else {
                std::cerr << "Warning: Matrix bounds exceeded for point " << i << " in frame " << k << std::endl;
            }
        }
    }

public:
    // Optimized M-step for rigid body transformation
    void M_rigidOptimized() {
        if (!isInitialized_) {
            std::cerr << "Error: SGBA not initialized!" << std::endl;
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process each frame in parallel
        #pragma omp parallel for
        for (int k = 0; k < K_; ++k) {
            computeFrameWeights(k);
        }
        
        // Update poses (sequential due to dependencies)
        updatePosesSequential();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Optimized M_rigid time: " << duration.count() << " ms" << std::endl;
    }
    
private:
    void computeFrameWeights(int k) {
        auto Zk = deqZ_.at(k);
        int numPoints = Zk->size();
        
        mW_.col(k).setZero();
        
        for (int j = 0; j < J_; ++j) {
            double lambdaTmp = 0;
            Eigen::Vector3d alphavSum = Eigen::Vector3d::Zero();
            
            for (int i = 0; i < numPoints; ++i) {
                int alphaIdx = i * J_ + j;
                if (alphaIdx < alphaMatrix_.cols()) {
                    double alpha = alphaMatrix_(k, alphaIdx);
                    
                    lambdaTmp += alpha;
                    
                    if (iterCnt_ >= TremainThd_) {
                        Eigen::Vector3d zki(Zk->points[i].x, Zk->points[i].y, Zk->points[i].z);
                        alphavSum += alpha * zki * vecSigma_.at(j);
                    }
                }
            }
            
            Lambda_(k, j) = lambdaTmp;
            b_(k, j) = lambdaTmp * vecSigma_.at(j);
            sumWeights(k) += lambdaTmp * vecSigma_.at(j);
            
            if (iterCnt_ >= TremainThd_) {
                W_.block<3, 1>(k * 3, j) = alphavSum;
                mW_.col(k) += alphavSum;
            }
        }
    }
    
    void updatePosesSequential() {
        if (iterCnt_ < TremainThd_) return;
        
        mX_ = M_ * b_.transpose();
        
        // Update poses using SVD (keep sequential for numerical stability)
        for (int k = 0; k < K_; ++k) {
            if (k == 0) continue; // Don't update reference frame
            
            Eigen::Vector3d mXk = mX_.col(k);
            Eigen::Vector3d mWk = mW_.col(k);
            
            Eigen::Matrix3d mat2Svd = M_ * W_.middleRows(k * 3, 3).transpose() 
                                    - mXk * mWk.transpose() / sumWeights(k);
            
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat2Svd, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            
            Eigen::Matrix3d Sk = Eigen::Matrix3d::Identity();
            Sk(2, 2) = (U * V.transpose()).determinant();
            
            Eigen::Matrix3d Rk = U * Sk * V.transpose();
            deqT_.at(k).block<3, 3>(0, 0) = Rk;
            deqT_.at(k).block<3, 1>(0, 3) = (mXk - Rk * mWk) / sumWeights(k);
        }
    }

public:
    // Optimized GMM parameter update
    void M_GMMOptimized() {
        if (!isInitialized_) {
            std::cerr << "Error: SGBA not initialized!" << std::endl;
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Parallel computation of cluster parameters
        #pragma omp parallel for
        for (int j = 0; j < J_; ++j) {
            updateClusterParameters(j);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Optimized M_GMM time: " << duration.count() << " ms" << std::endl;
    }
    
private:
    void updateClusterParameters(int j) {
        double alphaSum = 0;
        Eigen::Vector3d alphaZkSum = Eigen::Vector3d::Zero();
        double alphaNormSum = 0;
        
        // Accumulate over all frames and points
        for (int k = 0; k < K_; ++k) {
            auto Zk = deqZ_.at(k);
            int numPoints = Zk->size();
            
            Eigen::Matrix3d Rk = deqT_.at(k).block<3, 3>(0, 0);
            Eigen::Vector3d tk = deqT_.at(k).block<3, 1>(0, 3);
            
            for (int i = 0; i < numPoints; ++i) {
                int alphaIdx = i * J_ + j;
                if (alphaIdx < alphaMatrix_.cols()) {
                    double alpha = alphaMatrix_(k, alphaIdx);
                    
                    if (Zk->points[i].label == vecSemantic_.at(j)) {
                        alphaSum += alpha;
                        
                        Eigen::Vector3d zki(Zk->points[i].x, Zk->points[i].y, Zk->points[i].z);
                        Eigen::Vector3d transformedPoint = Rk * zki + tk;
                        
                        alphaZkSum += alpha * transformedPoint;
                        alphaNormSum += alpha * (transformedPoint - M_.col(j)).squaredNorm();
                    }
                }
            }
        }
        
        // Update cluster mean
        if (alphaSum > 0) {
            M_.col(j) = alphaZkSum / alphaSum;
            
            // Update cluster variance
            vecSigma_.at(j) = 1.0 / (alphaNormSum / (3.0 * alphaSum) + 0.0001);
            
            // Update mixing weight
            if (updateP_) {
                P_(j) = alphaSum / static_cast<double>(N_);
            }
        }
    }

public:
    // Memory pool for large matrices to avoid frequent allocations
    void preallocateMemory() {
        if (!isInitialized_) {
            std::cout << "Warning: SGBA not fully initialized, skipping preallocation" << std::endl;
            return;
        }

        // Estimate maximum sizes
        int maxPoints = 0;
        for (const auto& cloud : deqZ_) {
            if (cloud) {
                maxPoints += cloud->size();
            }
        }
        
        // Pre-allocate matrices
        if (J_ > 0 && maxPointsPerFrame_ > 0) {
            try {
                alphaMatrix_.resize(K_, maxPointsPerFrame_ * J_);
                betaMatrix_.resize(K_, maxPointsPerFrame_ * J_);
                clusterConstants_.resize(J_);
                sqrtSigmaCubed_.resize(J_);
                invTwoSigma_.resize(J_);
                
                std::cout << "Pre-allocated memory for " << maxPoints << " points and " 
                          << J_ << " clusters" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error in preallocation: " << e.what() << std::endl;
            }
        } else {
            std::cout << "Skipping preallocation due to invalid dimensions" << std::endl;
        }
    }

    // Spatial indexing for faster cluster assignment
    void buildSpatialIndex() {
        // Build KD-tree for cluster centers
        if (J_ > 10) {  // Only use spatial indexing for many clusters
            clusterKdTree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>());
            
            pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCenters(new pcl::PointCloud<pcl::PointXYZ>());
            for (int j = 0; j < J_; ++j) {
                pcl::PointXYZ center;
                center.x = M_(0, j);
                center.y = M_(1, j);
                center.z = M_(2, j);
                clusterCenters->push_back(center);
            }
            
            clusterKdTree_->setInputCloud(clusterCenters);
            useSpatialIndex_ = true;
        }
    }

public:
    // Public members (same as original SGBA)
    VoxelMap<PT> voxelMap_;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> vecCoZ_;
    std::vector<double> vecSigma_;
    std::vector<std::uint16_t> vecSemantic_;
    std::deque<typename pcl::PointCloud<PT>::Ptr> deqZ_;
    std::deque<Eigen::Matrix<double, 3, 4>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 4>>> deqT_;
    
    Eigen::Matrix<double, 3, Eigen::Dynamic> M_;
    Eigen::MatrixXd Lambda_;
    Eigen::MatrixXd W_;
    Eigen::MatrixXd b_;
    Eigen::Matrix<double, 3, Eigen::Dynamic> mW_;
    Eigen::Matrix<double, 3, Eigen::Dynamic> mX_;
    Eigen::VectorXd P_;
    Eigen::VectorXd sumWeights;
    
    size_t J_, K_, N_;  // J: number of landmarks/clusters, K: number of scans/frames
    double gamma_, h_;
    bool tensor_yes_, updateP_;
    std::uint16_t iterCnt_, TremainThd_;

private:
    // Optimized data structures
    Eigen::MatrixXd alphaMatrix_;
    Eigen::MatrixXd betaMatrix_;
    std::vector<double> clusterConstants_;
    std::vector<double> sqrtSigmaCubed_;
    std::vector<double> invTwoSigma_;
    int maxPointsPerFrame_;
    
    // Spatial indexing
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr clusterKdTree_;
    bool useSpatialIndex_ = false;
    
    // Initialization tracking
    bool isInitialized_;

private:
    // Helper function to get human-readable label names
    std::string getLabelName(uint16_t label) const {
        static std::map<uint16_t, std::string> labelNames = {
            {0, "unlabeled"}, {1, "outlier"}, {10, "car"}, {11, "bicycle"}, {13, "bus"},
            {15, "motorcycle"}, {16, "on-rails"}, {18, "truck"}, {20, "other-vehicle"},
            {30, "person"}, {31, "bicyclist"}, {32, "motorcyclist"}, {40, "road"},
            {44, "parking"}, {48, "sidewalk"}, {49, "other-ground"}, {50, "building"},
            {51, "fence"}, {52, "other-structure"}, {60, "lane-marking"}, {70, "vegetation"},
            {71, "trunk"}, {72, "terrain"}, {80, "pole"}, {81, "traffic-sign"}, {99, "other-object"}
        };
        
        auto it = labelNames.find(label);
        if (it != labelNames.end()) {
            return it->second;
        }
        return "unknown";
    }
};

#endif // SEMANTIC_BA_SGBA_HPP 