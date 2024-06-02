#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../estimator.h"
#include "../parameters.h"
#include <fstream>
#include <memory>

class EstimatorPublisher
{
public:
    EstimatorPublisher(ros::NodeHandle &n);

    void PublishAll(const Estimator &estimator, const std_msgs::Header &header, const double &compute_time);
    void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header);
    void pubRelocalization(const Estimator &estimator);

    // unused
    void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

private:
    void PrintStatistics(const double &imu_camera_clock_offset, const Eigen::Vector3d translation_camera_to_imu[], const Matrix3d rotation_camera_to_imu[], const Vector3d &position, const Vector3d &linear_velocity, const double &compute_time);

    void PubOdometry(const Vector3d &position, const Eigen::Quaterniond &orientation, const Vector3d &linear_velocity,
                     const Vector3d &drift_correction_translation, const Matrix3d &drift_correction_rotation, const std_msgs::Header &header);

    void PubKeyPoses(const vector<Vector3d> &key_poses, const std_msgs::Header &header);

    void PubCameraPose(const Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, const std_msgs::Header &header);

    void PubPointCloud(const std::vector<Eigen::Vector3d> &point_clouds, const std::vector<Eigen::Vector3d> &margined_point_clouds, const std_msgs::Header &header);

    void PubTF(const Eigen::Vector3d &position,
               const Eigen::Quaterniond &orientation,
               const Eigen::Vector3d &translation_camera_to_imu,
               const Eigen::Quaterniond &rotation_camera_to_imu,
               const std_msgs::Header &header);

    void PubKeyframe(const Eigen::Vector3d &position,
                     const Eigen::Quaterniond &orientation,
                     const std::vector<Eigen::Vector3d> &point_clouds,
                     std::vector<std::vector<float>> &feature_2d_3d_matches, const double &timestamp_2_back);

    void UpdatePoseMessage(geometry_msgs::Pose &pose_msg, const Vector3d &position, const Eigen::Quaterniond &orientation);
    void UpdateTwistMessage(geometry_msgs::Twist twist_msg, const Eigen::Vector3d &velocity);

    CameraPoseVisualization caemra_pose_visualization_;
    CameraPoseVisualization keyframe_base_visualization_;

    ros::Publisher pub_odometry_, pub_latest_odometry_;
    ros::Publisher pub_path_, pub_relo_path_;
    ros::Publisher pub_point_cloud_, pub_margin_cloud_;
    ros::Publisher pub_key_poses_;
    ros::Publisher pub_relo_relative_pose_;
    ros::Publisher pub_camera_pose_;
    ros::Publisher pub_camera_pose_visual_;
    ros::Publisher pub_keyframe_pose_;
    ros::Publisher pub_keyframe_point_;
    ros::Publisher pub_extrinsic_;

    double sum_of_path_;
    Vector3d last_path_;

    // State Variables
    Eigen::Vector3d position_estimated_current_;
    Eigen::Quaterniond orientation_estimated_current_;
    Eigen::Vector3d linear_velocity_estimated_current_;
    Eigen::Vector3d imu_linear_acceleration_estimated_bias_;
    Eigen::Vector3d imu_angular_velocity_estimated_bias_;

    // Observed Variables
    Eigen::Vector3d linear_acceleration_current_;
    Eigen::Vector3d angular_velocity_current_;

    // Camera to IMU
    Eigen::Vector3d translation_cameras_to_imu_[NUM_OF_CAM];
    Eigen::Matrix3d rotation_cameras_to_imu_[NUM_OF_CAM];
    double imu_camera_clock_offset_;

    // Relocalisation Corrections

    Vector3d drift_correction_translation_;
    Matrix3d drift_correction_rotation_;

    // Keyframes
    vector<Vector3d> key_poses_;

    Eigen::Vector3d camera_position_in_world_frame_;
    Eigen::Matrix3d camera_orientation_in_world_frame_;
    // ROS msgs
    visualization_msgs::Marker key_poses_msg_;
    nav_msgs::Path path_msg_, relo_path_msg_;
};
