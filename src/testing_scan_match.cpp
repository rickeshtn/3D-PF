#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pclomp/ndt_omp.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <string>
#include <sstream>

// For visualization
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>

// Function to create a transformation matrix from a quaternion and translation
Eigen::Matrix4f createTransformFromQuaternion(float qw, float qx, float qy, float qz, float x, float y, float z) {
    Eigen::Quaternionf quat(qw, qx, qy, qz);
    quat.normalize(); // Ensure the quaternion is normalized
    Eigen::Matrix3f rotation = quat.toRotationMatrix();
    
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = rotation;
    transform(0,3) = x;
    transform(1,3) = y;
    transform(2,3) = z;
    return transform;
}

/**
 * @brief Computes the ratio of source points that are within a certain distance of the target cloud.
 * 
 * @param transformed_source The source point cloud after applying the current transformation.
 * @param target The target point cloud.
 * @param tree The KD-Tree built from the target cloud for efficient nearest neighbor search.
 * @param distance_threshold The maximum distance to consider a point as aligned.
 * @return float The ratio of aligned points.
 */
float computeAlignedRatio(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& transformed_source,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
    pcl::KdTreeFLANN<pcl::PointXYZ>& tree,
    float distance_threshold)
{
    int aligned_count = 0;
    std::vector<int> nearest_indices(1);
    std::vector<float> nearest_distances(1);

    float threshold_squared = distance_threshold * distance_threshold;

    for (const auto& point : transformed_source->points) {
        if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
            continue; // Skip invalid points
        }

        if (tree.nearestKSearch(point, 1, nearest_indices, nearest_distances) > 0) {
            if (nearest_distances[0] <= threshold_squared) { // Compare squared distances
                aligned_count++;
            }
        }
    }

    // Avoid division by zero
    if (transformed_source->points.empty()) {
        return 0.0f;
    }

    return static_cast<float>(aligned_count) / static_cast<float>(transformed_source->points.size());
}

/**
 * @brief Parses command-line arguments to extract initial pose parameters.
 * 
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param qw Reference to store quaternion w.
 * @param qx Reference to store quaternion x.
 * @param qy Reference to store quaternion y.
 * @param qz Reference to store quaternion z.
 * @param x Reference to store translation x.
 * @param y Reference to store translation y.
 * @param z Reference to store translation z.
 * @return true If initial pose parameters are provided.
 * @return false If initial pose parameters are not provided.
 */
bool parseInitialPose(int argc, char** argv, float& qw, float& qx, float& qy, float& qz, float& x, float& y, float& z) {
    // Expecting arguments in the form: --init_pose qw qx qy qz x y z
    for (int i = 1; i < argc - 7; ++i) {
        std::string arg = argv[i];
        if (arg == "--init_pose") {
            std::istringstream ss_qw(argv[i+1]);
            std::istringstream ss_qx(argv[i+2]);
            std::istringstream ss_qy(argv[i+3]);
            std::istringstream ss_qz(argv[i+4]);
            std::istringstream ss_x(argv[i+5]);
            std::istringstream ss_y(argv[i+6]);
            std::istringstream ss_z(argv[i+7]);
            if (!(ss_qw >> qw) || !(ss_qx >> qx) || !(ss_qy >> qy) || !(ss_qz >> qz) ||
                !(ss_x >> x) || !(ss_y >> y) || !(ss_z >> z)) {
                std::cerr << "Error: Invalid initial pose parameters." << std::endl;
                return false;
            }
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    // Check for minimum number of arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " source_cloud.pcd target_cloud.pcd [--init_pose qw qx qy qz x y z]" << std::endl;
        return -1;
    }

    // Parse input file paths
    std::string source_file = argv[1];
    std::string target_file = argv[2];
    std::cout << "Source file: " << source_file << ", Target file: " << target_file << std::endl;

    // Load source and target point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile(source_file, *source) == -1) {
        std::cerr << "Failed to load " << source_file << std::endl;
        return -1;
    }
    std::cout << "Loaded source cloud with " << source->size() << " points." << std::endl;

    if (pcl::io::loadPCDFile(target_file, *target) == -1) {
        std::cerr << "Failed to load " << target_file << std::endl;
        return -1;
    }
    std::cout << "Loaded target cloud with " << target->size() << " points." << std::endl;

    // Remove NaN points from target cloud
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*target, *target, indices);
    std::cout << "Target cloud after removing NaNs has " << target->size() << " points." << std::endl;

    // Build a KD-Tree for the target cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> tree;
    tree.setInputCloud(target);

    // Initialize NDT
    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setResolution(1.0); // Adjust based on your data
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setMaximumIterations(30);
    ndt.setInputTarget(target);
    ndt.setNumThreads(4);
    ndt.setNeighborhoodSearchMethod(pclomp::DIRECT7);

    // Define search ranges and steps for grid search (used only if no initial pose is provided)
    float x_range = 5.0f; // Maximum x displacement to search
    float y_range = 5.0f; // Maximum y displacement to search
    float z_range = 5.0f; // Maximum z displacement to search
    float roll_range = static_cast<float>(M_PI / 4);  // Roll rotation range
    float pitch_range = static_cast<float>(M_PI / 4); // Pitch rotation range
    float yaw_range = static_cast<float>(M_PI / 4);   // Yaw rotation range
    float step = 1.0f;         // Step size for translations
    float angle_step = static_cast<float>(M_PI / 18); // Step size for rotations (10 degrees)

    // Initialize best score and transformation
    float best_ratio = 0.0f;
    Eigen::Matrix4f best_transform = Eigen::Matrix4f::Identity();

    // Set the source cloud for NDT
    ndt.setInputSource(source);

    // Parameters for the custom score
    float distance_threshold = 1.0f; // Adjust based on your application's scale

    // Parse initial pose if provided
    float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
    float init_x = 0.0f, init_y = 0.0f, init_z = 0.0f;
    bool has_init_pose = parseInitialPose(argc, argv, qw, qx, qy, qz, init_x, init_y, init_z);

    if (has_init_pose) {
        std::cout << "Initial pose provided:" << std::endl;
        std::cout << "  Quaternion: qw=" << qw << ", qx=" << qx << ", qy=" << qy << ", qz=" << qz << std::endl;
        std::cout << "  Translation: x=" << init_x << ", y=" << init_y << ", z=" << init_z << std::endl;

        // Create transformation matrix from the initial pose
        Eigen::Matrix4f initial_guess = createTransformFromQuaternion(qw, qx, qy, qz, init_x, init_y, init_z);

        // Align using the initial guess
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
        ndt.align(*aligned, initial_guess);

        if (ndt.hasConverged()) {
            // Compute the custom ratio score
            float ratio = computeAlignedRatio(aligned, target, tree, distance_threshold);
            std::cout << "Alignment converged with ratio score " << ratio << " using initial pose." << std::endl;

            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_transform = ndt.getFinalTransformation();
                std::cout << "Best ratio score updated to " << best_ratio << " with initial pose." << std::endl;
            }
        } else {
            std::cerr << "Alignment did not converge with the initial pose." << std::endl;
        }
    } else {
        // Perform grid search as no initial pose is provided
        std::cout << "No initial pose provided. Starting grid search for initial alignment..." << std::endl;

        for (float x = -x_range; x <= x_range; x += step) {
            for (float y = -y_range; y <= y_range; y += step) {
                for (float z = -z_range; z <= z_range; z += step) {
                    for (float roll = -roll_range; roll <= roll_range; roll += angle_step) {
                        for (float pitch = -pitch_range; pitch <= pitch_range; pitch += angle_step) {
                            for (float yaw = -yaw_range; yaw <= yaw_range; yaw += angle_step) {
                                // Create a guess transformation
                                Eigen::Matrix4f guess = Eigen::Matrix4f::Identity();

                                // Rotation matrices around each axis
                                Eigen::Matrix3f rot_x;
                                rot_x = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

                                Eigen::Matrix3f rot_y;
                                rot_y = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

                                Eigen::Matrix3f rot_z;
                                rot_z = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());

                                // Combined rotation
                                Eigen::Matrix3f rotation = rot_z * rot_y * rot_x;

                                guess.block<3,3>(0,0) = rotation;
                                guess(0,3) = x;
                                guess(1,3) = y;
                                guess(2,3) = z;

                                // Align the source cloud with the current guess
                                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
                                ndt.align(*aligned, guess);

                                // Check if alignment has converged
                                if (ndt.hasConverged()) {
                                    // Compute the custom ratio score
                                    float ratio = computeAlignedRatio(aligned, target, tree, distance_threshold);
                                    std::cout << "Alignment converged with ratio score " << ratio
                                              << " for x: " << x << ", y: " << y << ", z: " << z
                                              << ", roll: " << roll << ", pitch: " << pitch << ", yaw: " << yaw << std::endl;

                                    // Update the best score and transformation if current ratio is better
                                    if (ratio > best_ratio) {
                                        best_ratio = ratio;
                                        best_transform = ndt.getFinalTransformation();
                                        std::cout << "New best ratio score: " << best_ratio << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::cout << "Grid search completed." << std::endl;
        std::cout << "Best initial ratio from grid search: " << best_ratio << std::endl;
        std::cout << "Best initial transform from grid search:\n" << best_transform << std::endl;
    }

    // Refine alignment from the best transformation found
    std::cout << "Refining alignment from best transform..." << std::endl;
    ndt.align(*source, best_transform);
    if (ndt.hasConverged()) {
        best_transform = ndt.getFinalTransformation();
        // Apply the transformation to the source cloud for ratio computation
        pcl::PointCloud<pcl::PointXYZ>::Ptr refined_aligned(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*source, *refined_aligned, best_transform);
        float refined_ratio = computeAlignedRatio(refined_aligned, target, tree, distance_threshold);
        std::cout << "Refined ratio: " << refined_ratio << std::endl;
        std::cout << "Refined transform:\n" << best_transform << std::endl;
    } else {
        std::cerr << "Refinement did not converge." << std::endl;
    }

    // Apply the best transformation to the source cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_aligned(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*source, *final_aligned, best_transform);
    std::cout << "Applied best transformation to source cloud." << std::endl;

    // Visualization
    std::cout << "Starting visualization..." << std::endl;
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("NDT Alignment Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Define color handlers for target and aligned source clouds
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 255, 0, 0); // Red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(final_aligned, 0, 255, 0); // Green

    // Add point clouds to the viewer
    viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target_cloud");
    viewer->addPointCloud<pcl::PointXYZ>(final_aligned, source_color, "source_cloud_aligned");

    // Set point size for better visibility
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source_cloud_aligned");

    // Add coordinate system for reference
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Main visualization loop
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }

    std::cout << "Visualization completed." << std::endl;
    return 0;
}
