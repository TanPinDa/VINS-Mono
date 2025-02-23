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
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include "CameraPoseVisualization.h"
#include <eigen3/Eigen/Dense>
#include "../estimator.h"
#include "../parameters.h"
#include <fstream>
#include <memory>

class EstimatorPublisher {
 public:
  EstimatorPublisher(ros::NodeHandle &n);
  void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q,
                         const Eigen::Vector3d &V,
                         const std_msgs::Header &header);

  void printStatistics(const Estimator &estimator, double t);

  void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

  void pubInitialGuess(const Estimator &estimator,
                       const std_msgs::Header &header);

  void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

  void pubCameraPose(const Estimator &estimator,
                     const std_msgs::Header &header);

  void pubPointCloud(const Estimator &estimator,
                     const std_msgs::Header &header);

  void pubTF(const Estimator &estimator, const std_msgs::Header &header);

  void pubKeyframe(const Estimator &estimator);

  void pubRelocalization(const Estimator &estimator);

 private:
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
};
