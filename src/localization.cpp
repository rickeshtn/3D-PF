#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <random>
#include <pcl_conversions/pcl_conversions.h>
#include <numeric>

struct Particle {
    double x, y, yaw;
    double weight;
};

class ParticleFilterLocalization {
public:
    ParticleFilterLocalization(ros::NodeHandle &nh, ros::NodeHandle &pnh) : nh_(nh), pnh_(pnh), initialized_(false) {
        // Parameters
        pnh_.param<std::string>("map_path", map_path_, "/home/rickeshtn/Projects/berlin_company_data/ros1_accumulated_cloud/map_Manual.pcd");
        pnh_.param<int>("num_particles", num_particles_, 1000); 
        pnh_.param<double>("resample_threshold", resample_threshold_, 0.75);
        pnh_.param<double>("init_x_std", init_x_std_, 1.0);
        pnh_.param<double>("init_y_std", init_y_std_, 1.0);
        pnh_.param<double>("init_yaw_std", init_yaw_std_, 0.1);
        pnh_.param<double>("voxel_size", voxel_size_, 0.5); 
        pnh_.param<double>("voxel_scan_size_", voxel_scan_size_, 0.001); 

        std::string csv_filename;
        pnh_.param<std::string>("pose_csv_file", csv_filename, "/tmp/pose_log.csv");
        pose_log_.open(csv_filename.c_str(), std::ios::out | std::ios::trunc);
        if (!pose_log_) {
            ROS_WARN("Failed to open pose log file: %s", csv_filename.c_str());
        } else {
            pose_log_ << "timestamp,x,y,yaw\n"; // Write header line
        }

        ROS_INFO_STREAM("parameters:");
        ROS_INFO_STREAM("map_path: " << map_path_);
        ROS_INFO_STREAM("num_particles: " << num_particles_);
        ROS_INFO_STREAM("resample_threshold: " << resample_threshold_);
        ROS_INFO_STREAM("init_x_std: " << init_x_std_);
        ROS_INFO_STREAM("init_y_std: " << init_y_std_);
        ROS_INFO_STREAM("init_yaw_std: " << init_yaw_std_);
        ROS_INFO_STREAM("voxel_size: " << voxel_size_);
        ROS_INFO_STREAM("voxel_scan_size_: " << voxel_scan_size_);
        ROS_INFO_STREAM("Loading map file: " << map_path_);

        pcl::PointCloud<pcl::PointXYZ> raw_map;
        if (pcl::io::loadPCDFile(map_path_, raw_map) == -1) {
            ros::shutdown();
        }


        {
            pcl::VoxelGrid<pcl::PointXYZ> sor;
            pcl::PointCloud<pcl::PointXYZ>::Ptr raw_map_ptr(new pcl::PointCloud<pcl::PointXYZ>(raw_map));
            sor.setInputCloud(raw_map_ptr);
            sor.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
            pcl::PointCloud<pcl::PointXYZ> filtered;
            sor.filter(filtered);

            map_cloud_ = filtered;
        }

        map_cloud_ptr_.reset(new pcl::PointCloud<pcl::PointXYZ>(map_cloud_));
        kdtree_.reset(new pcl::KdTreeFLANN<pcl::PointXYZ>());
        kdtree_->setInputCloud(map_cloud_ptr_);

        map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_cloud", 1);
        publishMapCloud();

        map_publish_timer_ = nh_.createTimer(ros::Duration(1.0), [this](const ros::TimerEvent&) {
                if (!map_cloud_.empty()) {
                    publishMapCloud();
                } else {
                    ROS_WARN("map_cloud_ is empty; cannot publish.");
                }
            });

        odom_sub_ = nh_.subscribe("/kiss/odometry", 15, &ParticleFilterLocalization::odomCallback_smoothed, this);
        scan_sub_ = nh_.subscribe("/radar_data_topic", 1, &ParticleFilterLocalization::scanCallback, this);
        pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("pf_pose", 25);

        path_pub_ = nh_.advertise<nav_msgs::Path>("pf_path", 1);
        path_msg_.header.frame_id = "map";

        std::random_device rd;
        rng_.seed(rd());
    }

private:


    void publishMapCloud() 
    {
        if (map_cloud_.empty()) {
            ROS_ERROR("map_cloud_ is empty! Cannot publish map cloud.");
            return;
        }

        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(map_cloud_, map_msg);

        // Ensure the frame matches your RViz fixed frame
        map_msg.header.frame_id = "map"; 
        map_msg.header.stamp = ros::Time::now();

        // Publish the map cloud
        map_pub_.publish(map_msg);

        ROS_INFO_STREAM("Published map cloud with " << map_cloud_.size() << " points.");
    }

    void odomCallback_smoothed(const nav_msgs::OdometryConstPtr &odom) {
        current_odom_ = *odom;
        double dt = (current_odom_.header.stamp - last_odom_.header.stamp).toSec(); // Time difference
        if (!initialized_) {
            double x = current_odom_.pose.pose.position.x;
            double y = current_odom_.pose.pose.position.y;
            double yaw = yawFromQuaternion(current_odom_.pose.pose.orientation);

            // Initialize EMA smoothed variables
            ema_x_ = x;
            ema_y_ = y;
            ema_yaw_ = yaw;

            std::normal_distribution<double> dist_x(x, init_x_std_);
            std::normal_distribution<double> dist_y(y, init_y_std_);
            std::normal_distribution<double> dist_yaw(yaw, init_yaw_std_);

            particles_.resize(num_particles_);
            for (int i = 0; i < num_particles_; i++) {
                particles_[i].x = dist_x(rng_);
                particles_[i].y = dist_y(rng_);
                particles_[i].yaw = dist_yaw(rng_);
                particles_[i].weight = 1.0 / num_particles_;
            }
            initialized_ = true;
            map_based_pose_.x = x;
            map_based_pose_.y = y;
            map_based_pose_.yaw = yaw;
            last_odom_ = current_odom_;
            ROS_INFO("Particle Filter initialized with %d particles.", num_particles_);
        } 
        else {
            if (dt <= 0.0) {
                ROS_WARN("Invalid or zero time difference (dt). Skipping odometry update.");
                return;
            }

            // Extract raw odometry values
            double raw_x = current_odom_.pose.pose.position.x;
            double raw_y = current_odom_.pose.pose.position.y;
            double raw_yaw = yawFromQuaternion(current_odom_.pose.pose.orientation);

            // Apply EMA smoothing
            ema_x_ = alpha_ * raw_x + (1 - alpha_) * ema_x_;
            ema_y_ = alpha_ * raw_y + (1 - alpha_) * ema_y_;
            ema_yaw_ = alpha_ * raw_yaw + (1 - alpha_) * ema_yaw_;
            ema_yaw_ = normalizeAngle(ema_yaw_); // Normalize the smoothed yaw

            // Calculate deltas based on smoothed values
            double dx = ema_x_ - last_smoothed_x_;
            double dy = ema_y_ - last_smoothed_y_;
            double dyaw = angleDiff(ema_yaw_, last_smoothed_yaw_);

            // Update map-based pose
            map_based_pose_.x += cos(map_based_pose_.yaw) * dx - sin(map_based_pose_.yaw) * dy;
            map_based_pose_.y += sin(map_based_pose_.yaw) * dx + cos(map_based_pose_.yaw) * dy;
            map_based_pose_.yaw += dyaw;
            map_based_pose_.yaw = normalizeAngle(map_based_pose_.yaw);

            // Update particles
            std::normal_distribution<double> dist_x(0.0, 0.1);
            std::normal_distribution<double> dist_y(0.0, 0.1);
            std::normal_distribution<double> dist_yaw(0.0, 0.05);

            for (auto &p : particles_) {
                p.x = map_based_pose_.x + dist_x(rng_);
                p.y = map_based_pose_.y + dist_y(rng_);
                p.yaw = map_based_pose_.yaw + dist_yaw(rng_);
                p.yaw = normalizeAngle(p.yaw);
            }

            // Store smoothed values as the last pose
            last_smoothed_x_ = ema_x_;
            last_smoothed_y_ = ema_y_;
            last_smoothed_yaw_ = ema_yaw_;

            last_odom_ = current_odom_; // Update last odometry
        }
    }

    void scanCallback(const sensor_msgs::PointCloud2ConstPtr &scan_msg) {
        auto start_time = ros::Time::now();
        if (!initialized_) return;

        // Convert scan to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr scan_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*scan_msg, *scan_cloud);

        pcl::VoxelGrid<pcl::PointXYZ> sor;
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>(*scan_cloud));
        sor.setInputCloud(raw_scan_ptr);
        sor.setLeafSize(voxel_scan_size_, voxel_scan_size_, voxel_scan_size_);
        pcl::PointCloud<pcl::PointXYZ> filtered_scan;
        sor.filter(filtered_scan);

        auto weight_start_time = ros::Time::now();
        // Update step: compute weights
        double total_weight = 0.0;
        for (auto &p : particles_) {
            Eigen::Matrix4f transform = poseToMatrix(p.x, p.y, p.yaw);
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_scan(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(filtered_scan, *transformed_scan, transform);

            double w = computeWeight(transformed_scan);
            p.weight = w;
            total_weight += w;
        }

        auto weight_end_time = ros::Time::now();

        if (total_weight > 0) {
            for (auto &p : particles_) {
                p.weight /= total_weight;
            }
        } else {
            for (auto &p : particles_) {
                p.weight = 1.0 / num_particles_;
            }
        }
        
        // Resampling
        double neff = 1.0 / std::accumulate(particles_.begin(), particles_.end(), 0.0,
            [](double sum, const Particle &pp){ return sum + pp.weight*pp.weight; });
        if (neff < resample_threshold_ * num_particles_) {
            resampleParticles();
        }

        // Compute inside map score
        // Use the best pose (weighted mean)
        double best_x, best_y, best_yaw;
        poseMean(best_x, best_y, best_yaw);
        Eigen::Matrix4f mean_transform = poseToMatrix(best_x, best_y, best_yaw);
        pcl::PointCloud<pcl::PointXYZ>::Ptr mean_transformed_scan(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(filtered_scan, *mean_transformed_scan, mean_transform);
        double inside_score = insideMapScore(mean_transformed_scan);
        ROS_INFO_STREAM("Inside map score: " << inside_score);

        // Publish pose & path
        publishPose();
    }

    double computeWeight(const pcl::PointCloud<pcl::PointXYZ>::Ptr &scan) {
        int count = 0;
        std::vector<int> indices(1);
        std::vector<float> sqr_dist(1);
        
        for (auto &sp : scan->points) {
            if (kdtree_->nearestKSearch(sp, 1, indices, sqr_dist) > 0) {
                if (sqr_dist[0] < 0.25) { // within 0.5m
                    count++;
                }
            }
        }
        return count + 1e-9;
    }

    // New function: computes how well the scan fits inside the map.
    // Returns fraction of scan points that have a nearest map point within 0.5m
    double insideMapScore(const pcl::PointCloud<pcl::PointXYZ>::Ptr &scan) {
        if (scan->empty()) return 0.0;

        int count = 0;
        std::vector<int> indices(1);
        std::vector<float> sqr_dist(1);
        for (auto &sp : scan->points) {
            if (kdtree_->nearestKSearch(sp, 1, indices, sqr_dist) > 0) {
                if (sqr_dist[0] < 0.25) { // within 0.5m
                    count++;
                }
            }
        }
        return (double)count / (double)scan->size();
    }

    void resampleParticles() {
        std::vector<double> cdf(num_particles_);
        cdf[0] = particles_[0].weight;
        for (int i = 1; i < num_particles_; i++) {
            cdf[i] = cdf[i-1] + particles_[i].weight;
        }

        std::vector<Particle> new_particles(num_particles_);
        double step = 1.0 / num_particles_;
        std::uniform_real_distribution<double> dist(0.0, step);
        double r = dist(rng_);

        int idx = 0;
        for (int i = 0; i < num_particles_; i++) {
            double u = r + i*step;
            while (u > cdf[idx]) {
                idx++;
                if (idx >= num_particles_) idx = num_particles_ - 1;
            }
            new_particles[i] = particles_[idx];
            new_particles[i].weight = 1.0 / num_particles_;
        }
        particles_ = new_particles;
    }

    void poseMean(double &x, double &y, double &yaw) {
        x = 0; y = 0; 
        double sx=0, sy=0;
        for (auto &p : particles_) {
            x += p.x * p.weight;
            y += p.y * p.weight;
            sx += cos(p.yaw)*p.weight;
            sy += sin(p.yaw)*p.weight;
        }
        yaw = atan2(sy, sx);
    }

    void publishPose() {
        double x, y, yaw;
        poseMean(x, y, yaw);

        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = "map"; 
        ps.header.stamp = ros::Time::now();
        ps.pose.position.x = x;
        ps.pose.position.y = y;
        ps.pose.position.z = 0;
        tf2::Quaternion q;
        q.setRPY(0,0,yaw);
        ps.pose.orientation = tf2::toMsg(q);

        if (pose_log_.is_open()) {
            pose_log_ << ps.header.stamp.toSec() << "," 
                      << x << "," 
                      << y << "," 
                      << yaw << "\n";
            pose_log_.flush();
        }

        pose_pub_.publish(ps);

        path_msg_.header.stamp = ps.header.stamp;
        path_msg_.poses.push_back(ps);
        path_pub_.publish(path_msg_);
    }

    static double yawFromQuaternion(const geometry_msgs::Quaternion &q) {
        tf2::Quaternion quat(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        return yaw;
    }

    static double angleDiff(double a, double b) {
        double d = a - b;
        while (d > M_PI) d -= 2*M_PI;
        while (d < -M_PI) d += 2*M_PI;
        return d;
    }

    static double normalizeAngle(double a) {
        while (a > M_PI) a -= 2*M_PI;
        while (a < -M_PI) a += 2*M_PI;
        return a;
    }

    Eigen::Matrix4f poseToMatrix(double x, double y, double yaw) {
        Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
        float c = cos(yaw), s = sin(yaw);
        mat(0,0) = c; mat(0,1) = -s; mat(1,0) = s; mat(1,1) = c;
        mat(0,3) = x; mat(1,3) = y;
        return mat;
    }

    ros::NodeHandle nh_, pnh_;
    ros::Subscriber odom_sub_, scan_sub_;
    ros::Publisher pose_pub_, path_pub_, map_pub_;

    nav_msgs::Path path_msg_;

    std::string map_path_;
    pcl::PointCloud<pcl::PointXYZ> map_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_ptr_;
    boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZ>> kdtree_;

    nav_msgs::Odometry current_odom_, last_odom_;
    bool initialized_;

    std::vector<Particle> particles_;
    int num_particles_;
    double resample_threshold_;
    double init_x_std_, init_y_std_, init_yaw_std_;
    double voxel_size_, voxel_scan_size_;

    std::mt19937 rng_;

    struct {
        double x, y, yaw;
    } map_based_pose_;

    bool debug_flag_ = false;

    // Smoothing parameters
    double ema_x_ = 0.0;
    double ema_y_ = 0.0;
    double ema_yaw_ = 0.0;
    const double alpha_ = 0.5;
    double last_smoothed_x_ = 0.0, last_smoothed_y_ = 0.0, last_smoothed_yaw_ = 0.0;

    // Kalman filter state variables (not used in current logic)
    double kalman_x_ = 0.0, kalman_y_ = 0.0, kalman_yaw_ = 0.0;
    double kalman_cov_x_ = 1.0, kalman_cov_y_ = 1.0, kalman_cov_yaw_ = 1.0;
    const double process_noise_ = 0.01;
    const double measurement_noise_ = 0.1;
    double last_raw_x_ = 0.0, last_raw_y_ = 0.0, last_raw_yaw_ = 0.0;
    ros::Timer map_publish_timer_;
    std::ofstream pose_log_; // CSV log file

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "particle_filter_localization");
    ros::NodeHandle nh, pnh("~");
    ParticleFilterLocalization pf(nh, pnh);
    ros::AsyncSpinner spinner(2); // run with multiple threads if desired
    spinner.start();

    ros::waitForShutdown();
    return 0;
}
