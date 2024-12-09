// File: src/map_indexer.cpp

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

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


class MapIndexer {

public:
    MapIndexer(ros::NodeHandle& nh, ros::NodeHandle& pnh): nh_(nh), pnh_(pnh) {
        
        pnh_.param<std::string>("map_topic", map_topic_, "/map_cloud");
        pnh_.param<double>("grid_size", grid_size_, 10.0); // meters
        pnh_.param<double>("segment_size", segment_size_, 5.0); // meters
        pnh_.param<double>("fpfh_radius", fpfh_radius_, 1.0); // meters

        // Subscriber
        map_sub_ = nh_.subscribe(map_topic_, 1, &MapIndexer::mapCallback, this);

        // Initialize
        indexed_ = false;
    }

private:

    void mapCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        if (indexed_) {
            ROS_INFO("Map already indexed. Ignoring additional map messages.");
            return;
        }

        // Convert ROS message to PCL
        PointCloudT::Ptr map_cloud(new PointCloudT());
        pcl::fromROSMsg(*cloud_msg, *map_cloud);

        // Downsample the map for efficiency
        pcl::VoxelGrid<PointT> voxel_grid;
        voxel_grid.setInputCloud(map_cloud);
        voxel_grid.setLeafSize(0.5f, 0.5f, 0.5f); // Adjust as needed
        PointCloudT::Ptr filtered_map(new PointCloudT());
        voxel_grid.filter(*filtered_map);
        ROS_INFO("Downsampled map from %zu to %zu points.", map_cloud->size(), filtered_map->size());
        
    
    }

}



















int main(int argc, char** argv) {
    ros::init(argc, argv, "map_indexer");
    ros::NodeHandle nh, pnh("~");
    MapIndexer indexer(nh, pnh);
    ros::spin();
    return 0;
}
