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
    void printStatistics(const double &imu_camera_clock_offset, const Eigen::Vector3d translation_camera_to_imu[], const Matrix3d rotation_camera_to_imu[], const Vector3d &position, const Vector3d &linear_velocity, const double &compute_time);

    void pubOdometry(const Vector3d &position, const Eigen::Quaterniond orientation, const Vector3d &linear_velocity,
                                const Vector3d &drift_correction_translation, const Matrix3d &drift_correction_rotation, const std_msgs::Header &header);

    void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

    void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

    void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

    void pubTF(const Estimator &estimator, const std_msgs::Header &header);

    void pubKeyframe(const Estimator &estimator);

    void UpdatePoseMessage(geometry_msgs::Pose pose_msg, const Vector3d &position, const Eigen::Quaterniond orientation);
    void UpdateTwistMessage(geometry_msgs::Twist twist_msg, const Eigen::Vector3d &velocity);

    CameraPoseVisualization cameraposevisual;
    CameraPoseVisualization keyframebasevisual;
    ros::Publisher pub_odometry, pub_latest_odometry;
    ros::Publisher pub_path, pub_relo_path;
    ros::Publisher pub_point_cloud, pub_margin_cloud;
    ros::Publisher pub_key_poses;
    ros::Publisher pub_relo_relative_pose;
    ros::Publisher pub_camera_pose;
    ros::Publisher pub_camera_pose_visual;
    nav_msgs::Path path, relo_path;
    ros::Publisher pub_keyframe_pose;
    ros::Publisher pub_keyframe_point;
    ros::Publisher pub_extrinsic;
    double sum_of_path;

    Vector3d last_path;

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
    // ROS msgs

};
