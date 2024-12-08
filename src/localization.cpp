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
    double x, y, z;            // 3D position
    double roll, pitch, yaw;   // 3D orientation
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

        map_publish_timer_ = nh_.createTimer(ros::Duration(5.0), [this](const ros::TimerEvent&) {
                if (!map_cloud_.empty()) {
                    publishMapCloud();
                } else {
                    ROS_WARN("map_cloud_ is empty; cannot publish.");
                }
            });

        //odom_sub_ = nh_.subscribe("/kiss/odometry", 15, &ParticleFilterLocalization::odomCallback_smoothed, this);
        odom_sub_ = nh_.subscribe("/kiss/odometry", 15, &ParticleFilterLocalization::odomCallback, this);
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
            double z = current_odom_.pose.pose.position.z; // Extract Z position
            double roll, pitch, yaw;
            tf2::Matrix3x3(tf2::Quaternion(
                current_odom_.pose.pose.orientation.x,
                current_odom_.pose.pose.orientation.y,
                current_odom_.pose.pose.orientation.z,
                current_odom_.pose.pose.orientation.w)).getRPY(roll, pitch, yaw); // Extract 3D orientation

            // Initialize EMA smoothed variables
            ema_x_ = x;
            ema_y_ = y;
            ema_z_ = z; // Initialize Z smoothing
            ema_roll_ = roll;
            ema_pitch_ = pitch;
            ema_yaw_ = yaw;

            std::normal_distribution<double> dist_x(x, init_x_std_);
            std::normal_distribution<double> dist_y(y, init_y_std_);
            std::normal_distribution<double> dist_z(z, init_z_std_); // Initialize Z noise
            std::normal_distribution<double> dist_roll(roll, init_roll_std_);
            std::normal_distribution<double> dist_pitch(pitch, init_pitch_std_);
            std::normal_distribution<double> dist_yaw(yaw, init_yaw_std_);

            particles_.resize(num_particles_);
            for (int i = 0; i < num_particles_; i++) {
                particles_[i].x = dist_x(rng_);
                particles_[i].y = dist_y(rng_);
                particles_[i].z = dist_z(rng_);
                particles_[i].roll = dist_roll(rng_);
                particles_[i].pitch = dist_pitch(rng_);
                particles_[i].yaw = dist_yaw(rng_);
                particles_[i].weight = 1.0 / num_particles_;
            }
            initialized_ = true;
            map_based_pose_.x = x;
            map_based_pose_.y = y;
            map_based_pose_.z = z;
            map_based_pose_.roll = roll;
            map_based_pose_.pitch = pitch;
            map_based_pose_.yaw = yaw;
            last_odom_ = current_odom_;
            ROS_INFO("Particle Filter initialized with %d particles.", num_particles_);
        } else {
            if (dt <= 0.0) {
                ROS_WARN("Invalid or zero time difference (dt). Skipping odometry update.");
                return;
            }

            // Extract raw odometry values
            double raw_x = current_odom_.pose.pose.position.x;
            double raw_y = current_odom_.pose.pose.position.y;
            double raw_z = current_odom_.pose.pose.position.z;
            double raw_roll, raw_pitch, raw_yaw;
            tf2::Matrix3x3(tf2::Quaternion(
                current_odom_.pose.pose.orientation.x,
                current_odom_.pose.pose.orientation.y,
                current_odom_.pose.pose.orientation.z,
                current_odom_.pose.pose.orientation.w)).getRPY(raw_roll, raw_pitch, raw_yaw);

            // Apply EMA smoothing
            ema_x_ = alpha_ * raw_x + (1 - alpha_) * ema_x_;
            ema_y_ = alpha_ * raw_y + (1 - alpha_) * ema_y_;
            ema_z_ = alpha_ * raw_z + (1 - alpha_) * ema_z_;
            ema_roll_ = alpha_ * raw_roll + (1 - alpha_) * ema_roll_;
            ema_pitch_ = alpha_ * raw_pitch + (1 - alpha_) * ema_pitch_;
            ema_yaw_ = alpha_ * raw_yaw + (1 - alpha_) * ema_yaw_;
            ema_yaw_ = normalizeAngle(ema_yaw_); // Normalize the smoothed yaw

            // Calculate deltas based on smoothed values
            double dx = ema_x_ - last_smoothed_x_;
            double dy = ema_y_ - last_smoothed_y_;
            double dz = ema_z_ - last_smoothed_z_;
            double droll = ema_roll_ - last_smoothed_roll_;
            double dpitch = ema_pitch_ - last_smoothed_pitch_;
            double dyaw = angleDiff(ema_yaw_, last_smoothed_yaw_);

            // Update map-based pose
            map_based_pose_.x += dx;
            map_based_pose_.y += dy;
            map_based_pose_.z += dz;
            map_based_pose_.roll += droll;
            map_based_pose_.pitch += dpitch;
            map_based_pose_.yaw += dyaw;
            map_based_pose_.yaw = normalizeAngle(map_based_pose_.yaw);

            // Update particles for 3D motion
            std::normal_distribution<double> dist_x(0.0, 0.1);
            std::normal_distribution<double> dist_y(0.0, 0.1);
            std::normal_distribution<double> dist_z(0.0, 0.1);  
            std::normal_distribution<double> dist_roll(0.0, 0.05);
            std::normal_distribution<double> dist_pitch(0.0, 0.05);
            std::normal_distribution<double> dist_yaw(0.0, 0.05);

            for (auto &p : particles_) {
                // Update position
                p.x += dx + dist_x(rng_);
                p.y += dy + dist_y(rng_);
                p.z += dz + dist_z(rng_);
                
                // Update orientation
                p.roll += droll + dist_roll(rng_);
                p.pitch += dpitch + dist_pitch(rng_);
                p.yaw += dyaw + dist_yaw(rng_);
                p.yaw = normalizeAngle(p.yaw); 
            }

            // Store smoothed values as the last pose
            last_smoothed_x_ = ema_x_;
            last_smoothed_y_ = ema_y_;
            last_smoothed_z_ = ema_z_;
            last_smoothed_roll_ = ema_roll_;
            last_smoothed_pitch_ = ema_pitch_;
            last_smoothed_yaw_ = ema_yaw_;

            last_odom_ = current_odom_; // Update last odometry
        }
    }

    void odomCallback(const nav_msgs::OdometryConstPtr &odom) 
    {
        current_odom_ = *odom;
        double dt = (current_odom_.header.stamp - last_odom_.header.stamp).toSec(); // Time difference

        if (!initialized_) {
            double x = current_odom_.pose.pose.position.x;
            double y = current_odom_.pose.pose.position.y;
            double z = current_odom_.pose.pose.position.z;

            double roll, pitch, yaw;
            tf2::Matrix3x3(tf2::Quaternion(
                current_odom_.pose.pose.orientation.x,
                current_odom_.pose.pose.orientation.y,
                current_odom_.pose.pose.orientation.z,
                current_odom_.pose.pose.orientation.w)).getRPY(roll, pitch, yaw);

            // Initialize particles
            std::normal_distribution<double> dist_x(x, init_x_std_);
            std::normal_distribution<double> dist_y(y, init_y_std_);
            std::normal_distribution<double> dist_z(z, init_z_std_);
            std::normal_distribution<double> dist_roll(roll, init_roll_std_);
            std::normal_distribution<double> dist_pitch(pitch, init_pitch_std_);
            std::normal_distribution<double> dist_yaw(yaw, init_yaw_std_);

            particles_.resize(num_particles_);
            for (int i = 0; i < num_particles_; i++) {
                particles_[i].x = dist_x(rng_);
                particles_[i].y = dist_y(rng_);
                particles_[i].z = dist_z(rng_);
                particles_[i].roll = dist_roll(rng_);
                particles_[i].pitch = dist_pitch(rng_);
                particles_[i].yaw = dist_yaw(rng_);
                particles_[i].weight = 1.0 / num_particles_;
            }

            // Initialize map-based pose
            map_based_pose_.x = x;
            map_based_pose_.y = y;
            map_based_pose_.z = z;
            map_based_pose_.roll = roll;
            map_based_pose_.pitch = pitch;
            map_based_pose_.yaw = yaw;

            initialized_ = true;
            last_odom_ = current_odom_;
            ROS_INFO("Particle Filter initialized with %d particles.", num_particles_);
        } 
        else {
            if (dt <= 0.0) {
                ROS_WARN("Invalid or zero time difference (dt). Skipping odometry update.");
                return;
            }

            // Extract deltas from odometry
            double dx = current_odom_.pose.pose.position.x - last_odom_.pose.pose.position.x;
            double dy = current_odom_.pose.pose.position.y - last_odom_.pose.pose.position.y;
            double dz = current_odom_.pose.pose.position.z - last_odom_.pose.pose.position.z;

            double roll, pitch, yaw;
            tf2::Matrix3x3(tf2::Quaternion(
                current_odom_.pose.pose.orientation.x,
                current_odom_.pose.pose.orientation.y,
                current_odom_.pose.pose.orientation.z,
                current_odom_.pose.pose.orientation.w)).getRPY(roll, pitch, yaw);

            double last_roll, last_pitch, last_yaw;
            tf2::Matrix3x3(tf2::Quaternion(
                last_odom_.pose.pose.orientation.x,
                last_odom_.pose.pose.orientation.y,
                last_odom_.pose.pose.orientation.z,
                last_odom_.pose.pose.orientation.w)).getRPY(last_roll, last_pitch, last_yaw);

            double droll = roll - last_roll;
            double dpitch = pitch - last_pitch;
            double dyaw = angleDiff(yaw, last_yaw);

            // Calculate speed and angular rates
            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            double speed = distance / dt;
            double turn_rate = sqrt(droll * droll + dpitch * dpitch + dyaw * dyaw) / dt;

            if ((speed < 0.05 || speed > 30.0) || (turn_rate > 10.0)) {
                ROS_WARN("Irregular speed or turn rate. Skipping odometry update.");
                return;
            }

            // Update map-based pose
            map_based_pose_.x += dx;
            map_based_pose_.y += dy;
            map_based_pose_.z += dz;
            map_based_pose_.roll += droll;
            map_based_pose_.pitch += dpitch;
            map_based_pose_.yaw += dyaw;
            map_based_pose_.yaw = normalizeAngle(map_based_pose_.yaw);

            // Update particles
            std::normal_distribution<double> dist_x(0.0, 0.1);
            std::normal_distribution<double> dist_y(0.0, 0.1);
            std::normal_distribution<double> dist_z(0.0, 0.1);
            std::normal_distribution<double> dist_roll(0.0, 0.05);
            std::normal_distribution<double> dist_pitch(0.0, 0.05);
            std::normal_distribution<double> dist_yaw(0.0, 0.05);

            for (auto &p : particles_) {
                p.x += dx + dist_x(rng_);
                p.y += dy + dist_y(rng_);
                p.z += dz + dist_z(rng_);
                p.roll += droll + dist_roll(rng_);
                p.pitch += dpitch + dist_pitch(rng_);
                p.yaw += dyaw + dist_yaw(rng_);
                p.yaw = normalizeAngle(p.yaw);
            }

            // Update last odometry
            last_odom_ = current_odom_;
        }
    }



    void reinitializeParticles() {
        double x = current_odom_.pose.pose.position.x;
        double y = current_odom_.pose.pose.position.y;
        double z = current_odom_.pose.pose.position.z;

        double roll, pitch, yaw;
        tf2::Matrix3x3(tf2::Quaternion(
            current_odom_.pose.pose.orientation.x,
            current_odom_.pose.pose.orientation.y,
            current_odom_.pose.pose.orientation.z,
            current_odom_.pose.pose.orientation.w)).getRPY(roll, pitch, yaw);

        // Reinitialize particles with a larger spread
        std::normal_distribution<double> dist_x(x, 5.0); // Wider spread
        std::normal_distribution<double> dist_y(y, 5.0); // Wider spread
        std::normal_distribution<double> dist_z(z, 2.0); // Wider spread
        std::normal_distribution<double> dist_roll(roll, M_PI / 6); // 30 degrees spread
        std::normal_distribution<double> dist_pitch(pitch, M_PI / 6);
        std::normal_distribution<double> dist_yaw(yaw, M_PI / 4); // 45 degrees spread

        particles_.resize(num_particles_);
        for (int i = 0; i < num_particles_; i++) {
            particles_[i].x = dist_x(rng_);
            particles_[i].y = dist_y(rng_);
            particles_[i].z = dist_z(rng_);
            particles_[i].roll = dist_roll(rng_);
            particles_[i].pitch = dist_pitch(rng_);
            particles_[i].yaw = dist_yaw(rng_);
            particles_[i].weight = 1.0 / num_particles_;
        }

        // Reset map-based pose
        map_based_pose_.x = x;
        map_based_pose_.y = y;
        map_based_pose_.z = z;
        map_based_pose_.roll = roll;
        map_based_pose_.pitch = pitch;
        map_based_pose_.yaw = yaw;

        ROS_WARN("Reinitialized particles and map-based pose.");
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
            // Create transformation based on particle pose
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.translation() << p.x, p.y, p.z;
            transform.rotate(Eigen::AngleAxisf(p.roll, Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(p.pitch, Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(p.yaw, Eigen::Vector3f::UnitZ()));

            // Transform scan points
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_scan(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(filtered_scan, *transformed_scan, transform);

            // Compute weight for transformed scan
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
        double best_x, best_y, best_z, best_roll, best_pitch, best_yaw;
        poseMean(best_x, best_y, best_z, best_roll, best_pitch, best_yaw);
        Eigen::Affine3f mean_transform = Eigen::Affine3f::Identity();
        mean_transform.translation() << best_x, best_y, best_z;
        mean_transform.rotate(Eigen::AngleAxisf(best_roll, Eigen::Vector3f::UnitX()) *
                            Eigen::AngleAxisf(best_pitch, Eigen::Vector3f::UnitY()) *
                            Eigen::AngleAxisf(best_yaw, Eigen::Vector3f::UnitZ()));

        pcl::PointCloud<pcl::PointXYZ>::Ptr mean_transformed_scan(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(filtered_scan, *mean_transformed_scan, mean_transform);
        double inside_score = insideMapScore(mean_transformed_scan);

        if (inside_score == 0.0) {
            ROS_WARN("Inside map score is 0. Reinitializing particles.");
            reinitializeParticles();
            return;
        }

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

    void poseMean(double &x, double &y, double &z, double &roll, double &pitch, double &yaw) {
        x = 0.0; 
        y = 0.0; 
        z = 0.0;
        double sx = 0.0, sy = 0.0;
        roll = 0.0; 
        pitch = 0.0;
        double total_weight = 0.0;

        for (const auto &p : particles_) {
            x += p.x * p.weight;
            y += p.y * p.weight;
            z += p.z * p.weight;
            roll += p.roll * p.weight;
            pitch += p.pitch * p.weight;
            sx += cos(p.yaw) * p.weight;
            sy += sin(p.yaw) * p.weight;
            total_weight += p.weight;
        }

        if (total_weight > 0) {
            x /= total_weight;
            y /= total_weight;
            z /= total_weight;
            roll /= total_weight;
            pitch /= total_weight;
            yaw = atan2(sy, sx);
        } else {
            ROS_WARN("Total weight of particles is zero during pose mean calculation.");
        }
    }


    void publishPose() {
        double x = 0.0, y = 0.0, z = 0.0;
        double roll = 0.0, pitch = 0.0, yaw = 0.0;

        // Compute weighted mean for pose from particles
        double total_weight = 0.0;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        double sroll = 0.0, spitch = 0.0, syaw = 0.0;

        for (const auto &p : particles_) {
            x += p.x * p.weight;
            y += p.y * p.weight;
            z += p.z * p.weight;
            sx += cos(p.yaw) * p.weight;
            sy += sin(p.yaw) * p.weight;
            sroll += p.roll * p.weight;
            spitch += p.pitch * p.weight;
            total_weight += p.weight;
        }

        if (total_weight > 0) {
            x /= total_weight;
            y /= total_weight;
            z /= total_weight;
            roll = sroll / total_weight;
            pitch = spitch / total_weight;
            yaw = atan2(sy, sx);
        }

        // Publish pose as a ROS PoseStamped message
        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = "map";
        ps.header.stamp = ros::Time::now();
        ps.pose.position.x = x;
        ps.pose.position.y = y;
        ps.pose.position.z = z;

        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        ps.pose.orientation = tf2::toMsg(q);

        // Log pose to CSV if enabled
        if (pose_log_.is_open()) {
            pose_log_ << ps.header.stamp.toSec() << "," 
                    << x << "," 
                    << y << "," 
                    << z << "," 
                    << roll << "," 
                    << pitch << "," 
                    << yaw << "\n";
            pose_log_.flush();
        }

        // Publish pose and update path
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
    double init_x_std_, init_y_std_, init_yaw_std_,init_z_std_, init_roll_std_, init_pitch_std_;
    double voxel_size_, voxel_scan_size_;

    std::mt19937 rng_;

    struct {
        double x, y, yaw, z, roll, pitch;
    } map_based_pose_;

    bool debug_flag_ = false;

    // Smoothing parameters
    double ema_x_ = 0.0;
    double ema_y_ = 0.0;
    double ema_yaw_ = 0.0;
    double ema_z_ = 0.0;
    double ema_roll_ = 0.0;
    double ema_pitch_ = 0.0;

    const double alpha_ = 0.5;
    double last_smoothed_x_ = 0.0, last_smoothed_y_ = 0.0, last_smoothed_yaw_ = 0.0, last_smoothed_z_ = 0.0, last_smoothed_roll_ = 0.0, last_smoothed_pitch_ = 0.0;

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
