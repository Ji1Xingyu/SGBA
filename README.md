# SGBA: Semantic Gaussian Mixture Model-Based LiDAR Bundle Adjustment

This repository implements SGBA, a LiDAR bundle adjustment system that uses semantic Gaussian mixture models for robust pose estimation.

## Quick Start

### Prerequisites
- ROS (tested with ROS Noetic)
- PCL
- Eigen3
- OpenMP

### Build
```bash
cd ~/catkin_ws
catkin build semantic_ba
```

### Configuration
Edit `config/read_label.yaml` to set the test mode and parameters.

## Running Tests

### 1. Virtual Planes Test
Test with synthetic planar data for algorithm validation.

**Setup:**
1. Set test mode in `config/read_label.yaml`:
   ```yaml
   test_mode: "virtual_planes"
   ```

2. Configure virtual plane parameters:
   ```yaml
   virtual_planes:
     pcd_file: "/path/to/your/virtual_plane_data.pcd"
     max_frames: 20
     use_identity_transforms: true
   ```

**Run:**
```bash
rosrun semanticBA algoTest
```

### 2. KITTI Semantic Test
Test with real KITTI semantic dataset.

**Setup:**
1. Download KITTI semantic dataset
2. Set test mode in `config/read_label.yaml`:
   ```yaml
   test_mode: "kitti"
   ```

3. Configure KITTI parameters:
   ```yaml
   semantic_kitti:
     kittiPath: "/path/to/kitti_semantic/sequence"
     frameCnt: 400      # starting frame
     winSize: 20        # sliding window size
     label2Read: [80, 71, 40, 48]  # semantic labels to use
   ```

**Run:**
```bash
rosrun semanticBA algoTest
```

### Controls
- Press **'n'** to proceed to next iteration
- Press **'s'** to show performance statistics  
- Press **'q'** to quit

## Dataset Format
- **Virtual Planes**: PCD files with intensity (frame ID) and curvature (virtual semantic label)
- **KITTI**: Standard KITTI semantic format with point-wise semantic labels

Video: \
[![Watch the video](https://img.youtube.com/vi/_au_uUMCUpU/hqdefault.jpg)](https://youtu.be/_au_uUMCUpU)
