// File: src/MapIndexer.cpp

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/fpfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

class MapIndexer {

public:
    MapIndexer(ros::NodeHandle& nh, ros::NodeHandle& pnh): nh_(nh), pnh_(pnh) {
        
        pnh_.param<std::string>("map_topic", map_topic_, "/map_cloud");
        pnh_.param<double>("grid_size", grid_size_, 10.0); // meters
        pnh_.param<double>("segment_size", segment_size_, 5.0); // meters
        pnh_.param<double>("fpfh_radius", fpfh_radius_, 1.0); // meters

        ROS_INFO("Initializing MapIndexer Node...");
        ROS_INFO("Subscribed to map_topic: %s", map_topic_.c_str());
        ROS_INFO("Grid size: %f meters", grid_size_);
        ROS_INFO("Segment size: %f meters", segment_size_);
        ROS_INFO("FPFH radius: %f meters", fpfh_radius_);

        // Subscriber
        map_sub_ = nh_.subscribe(map_topic_, 1, &MapIndexer::mapCallback, this);

        // Initialize
        indexed_ = false;
    }

private:

    void mapCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        ROS_INFO("Received map message.");

        if (indexed_) {
            ROS_WARN("Map already indexed. Ignoring additional map messages.");
            return;
        }

        // Convert ROS message to PCL
        PointCloudT::Ptr map_cloud(new PointCloudT());
        ROS_INFO("Converting ROS PointCloud2 message to PCL PointCloud...");
        pcl::fromROSMsg(*cloud_msg, *map_cloud);
        ROS_INFO("Converted map cloud has %zu points.", map_cloud->size());

        // Downsample the map for efficiency
        pcl::VoxelGrid<PointT> voxel_grid;
        voxel_grid.setInputCloud(map_cloud);
        voxel_grid.setLeafSize(0.5f, 0.5f, 0.5f); // Adjust as needed
        PointCloudT::Ptr filtered_map(new PointCloudT());
        ROS_INFO("Downsampling the map...");
        voxel_grid.filter(*filtered_map);
        ROS_INFO("Downsampled map from %zu to %zu points.", map_cloud->size(), filtered_map->size());

        // Segment the map into grid cells
        std::vector<PointCloudT::Ptr> segments;
        ROS_INFO("Segmenting the map into grid cells...");
        segmentMap(filtered_map, segments);
        ROS_INFO("Segmented map into %zu segments.", segments.size());

        if (segments.empty()) {
            ROS_WARN("No segments were created. Check grid and segment sizes.");
            return;
        }

        // Compute descriptors for each segment
        std::vector<std::vector<float>> descriptors;
        descriptors.reserve(segments.size());

        ROS_INFO("Computing FPFH descriptors for each segment...");
        for (size_t i = 0; i < segments.size(); ++i) {
            ROS_DEBUG("Computing descriptor for segment %zu...", i);
            std::vector<float> desc = computeFPFHDescriptor(segments[i]);
            if (desc.empty()) {
                ROS_WARN("Descriptor computation failed for segment %zu.", i);
                continue;
            }
            descriptors.push_back(desc);
            ROS_DEBUG("Computed FPFH descriptor for segment %zu.", i);
        }
        ROS_INFO("Computed descriptors for %zu segments.", descriptors.size());

        if (descriptors.empty()) {
            ROS_ERROR("No descriptors were computed. Aborting indexing.");
            return;
        }

        // Build the index (KD-Tree for FPFH descriptors)
        ROS_INFO("Building KD-Tree index for FPFH descriptors...");
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::FPFHSignature33>());
        for (size_t i = 0; i < descriptors.size(); ++i) {
            pcl::FPFHSignature33 point;
            for (int j = 0; j < 33; ++j) {
                point.histogram[j] = (j < descriptors[i].size()) ? descriptors[i][j] : 0.0f;
            }
            fpfh_cloud->points.push_back(point);
        }

        ROS_INFO("KD-Tree construction started.");
        pcl::KdTreeFLANN<pcl::FPFHSignature33> descriptor_kdtree;
        descriptor_kdtree.setInputCloud(fpfh_cloud);
        ROS_INFO("KD-Tree index built successfully.");

        // Save the index and descriptors to file for later use
        ROS_INFO("Saving descriptors to file 'map_index.bin'...");
        saveIndexToFile(descriptors, "map_index.bin");
        ROS_INFO("Descriptors saved to 'map_index.bin'.");

        indexed_ = true;
        ROS_INFO("Map indexing completed.");
    }

    void segmentMap(const PointCloudT::Ptr& map_cloud, std::vector<PointCloudT::Ptr>& segments) {
        ROS_INFO("Determining map boundaries...");
        // Define grid boundaries
        PointT min_pt, max_pt;
        pcl::getMinMax3D<PointT>(*map_cloud, min_pt, max_pt);
        ROS_INFO("Map boundaries - Min: (%f, %f, %f), Max: (%f, %f, %f)",
                 min_pt.x, min_pt.y, min_pt.z,
                 max_pt.x, max_pt.y, max_pt.z);

        double min_x = min_pt.x;
        double max_x = max_pt.x;
        double min_y = min_pt.y;
        double max_y = max_pt.y;
        double min_z = min_pt.z;
        double max_z = max_pt.z;

        int grid_x = static_cast<int>((max_x - min_x) / grid_size_) + 1;
        int grid_y = static_cast<int>((max_y - min_y) / grid_size_) + 1;

        ROS_INFO("Grid dimensions - X: %d cells, Y: %d cells", grid_x, grid_y);

        // Initialize grid cells
        ROS_INFO("Initializing grid cells...");
        std::vector<PointCloudT::Ptr> grid_segments(grid_x * grid_y, nullptr);
        for (auto& seg : grid_segments) {
            seg.reset(new PointCloudT());
        }

        // Assign points to grid cells
        ROS_INFO("Assigning points to grid cells...");
        for (const auto& point : map_cloud->points) {
            int ix = static_cast<int>((point.x - min_x) / grid_size_);
            int iy = static_cast<int>((point.y - min_y) / grid_size_);
            if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y) {
                int index = ix * grid_y + iy;
                grid_segments[index]->points.push_back(point);
            }
        }

        // Collect non-empty segments
        ROS_INFO("Collecting non-empty segments...");
        for (size_t i = 0; i < grid_segments.size(); ++i) {
            if (grid_segments[i]->points.empty()) continue;
            segments.push_back(grid_segments[i]);
        }
        ROS_INFO("Total non-empty segments collected: %zu", segments.size());
    }

    std::vector<float> computeFPFHDescriptor(const PointCloudT::Ptr& segment) {
        // Check if the segment has enough points
        if (segment->size() < 100) { // Thresholds to ignore sparse segments. 
            ROS_WARN("Segment has too few points (%zu). Skipping descriptor computation.", segment->size());
            return std::vector<float>();
        }

        // Compute normals
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        ne.setInputCloud(segment);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(1.0); 
        ROS_DEBUG("Computing normals for segment...");
        ne.compute(*normals);

        if (normals->empty()) {
            ROS_WARN("Normal estimation failed for segment.");
            return std::vector<float>();
        }

        // Compute FPFH descriptors
        pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33> fpfh;
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_desc(new pcl::PointCloud<pcl::FPFHSignature33>());
        fpfh.setInputCloud(segment);
        fpfh.setInputNormals(normals);
        fpfh.setSearchMethod(tree);
        fpfh.setRadiusSearch(fpfh_radius_);
        ROS_DEBUG("Computing FPFH descriptors for segment...");
        fpfh.compute(*fpfh_desc);

        if (fpfh_desc->empty()) {
            ROS_WARN("FPFH computation failed for segment.");
            return std::vector<float>();
        }

        // Aggregate FPFH descriptors (e.g., average)
        std::vector<float> descriptor(33, 0.0f);
        for (const auto& point : fpfh_desc->points) {
            for (int i = 0; i < 33; ++i) {
                descriptor[i] += point.histogram[i];
            }
        }
        if (!fpfh_desc->points.empty()) {
            for (auto& val : descriptor) {
                val /= static_cast<float>(fpfh_desc->points.size());
            }
        }

        return descriptor;
    }


    void saveIndexToFile(const std::vector<std::vector<float>>& descriptors, const std::string& filename) {
        ROS_INFO("Saving descriptors to binary file: %s", filename.c_str());
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            ROS_ERROR("Failed to open file %s for writing index.", filename.c_str());
            return;
        }

        // Save number of descriptors
        size_t num_desc = descriptors.size();
        ofs.write(reinterpret_cast<const char*>(&num_desc), sizeof(size_t));
        ROS_DEBUG("Number of descriptors saved: %zu", num_desc);

        // Save each descriptor
        for (size_t i = 0; i < descriptors.size(); ++i) {
            size_t size = descriptors[i].size();
            ofs.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
            ofs.write(reinterpret_cast<const char*>(descriptors[i].data()), size * sizeof(float));
            ROS_DEBUG("Saved descriptor %zu with size %zu", i, size);
        }

        ofs.close();
        ROS_INFO("Successfully saved descriptors to %s", filename.c_str());
    }

    ros::NodeHandle nh_, pnh_;
    ros::Subscriber map_sub_;
    std::string map_topic_;
    double grid_size_;
    double segment_size_;
    double fpfh_radius_;
    bool indexed_;

    std::shared_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;

};


int main(int argc, char** argv) {
    ros::init(argc, argv, "map_indexer");
    ros::NodeHandle nh, pnh("~");
    ROS_INFO("Starting MapIndexer Node...");
    MapIndexer indexer(nh, pnh);
    ros::spin();
    return 0;
}
