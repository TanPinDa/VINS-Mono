/**
 * @file pose_graph_node.hpp
 * @brief
 * @date 08-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef POSE_GRAPH_ROS_POSE_GRAPH_NODE_HPP
#define POSE_GRAPH_ROS_POSE_GRAPH_NODE_HPP

#include <memory>
#include <mutex>
#include <thread>
#include <queue>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "pose_graph/pose_graph.hpp"
#include "pose_graph/details/pose_graph_event_observer.hpp"
#include "pose_graph_ros/utility/CameraPoseVisualization.h"

namespace pose_graph {
class PoseGraphNode : public PoseGraphEventObserver {
 public:
  ~PoseGraphNode();

  bool Start();

 private:
  bool ReadParameters();
  void StartPublishersAndSubscribers();
  void StartCommandAndProcessingThreads();
  void Process();
  void Command();
  void Publish();
  void NewSequence();

  // PoseGraphEventObserver callbacks
  void OnPoseGraphLoaded() final;
  void OnPoseGraphSaved() final;
  void OnKeyFrameAdded(KeyFrame::Attributes kf_attribute) final;
  void OnKeyFrameLoaded(KeyFrame::Attributes kf_attribute, int count) final;
  void OnKeyFrameConnectionFound(KeyFrame::Attributes current_kf_attribute,
                                 KeyFrame::Attributes old_kf_attribute,
                                 std::vector<cv::Point2f> matched_2d_old_norm,
                                 std::vector<double> matched_id,
                                 cv::Mat& thumb_image) final;
  void OnPoseGraphOptimization(
      std::vector<KeyFrame::Attributes> kf_attributes) final;
  void OnNewSequentialEdge(Vector3d p1, Vector3d p2) final;
  void OnNewLoopEdge(Vector3d p1, Vector3d p2) final;

  // ros callbacks
  void ImuForwardCallback(const nav_msgs::OdometryConstPtr& forward_msg);
  void VioCallback(const nav_msgs::OdometryConstPtr& pose_msg);
  void ImageCallback(const sensor_msgs::ImageConstPtr& image_msg);
  void PoseCallback(const nav_msgs::OdometryConstPtr& pose_msg);
  void ExtrinsicCallback(const nav_msgs::OdometryConstPtr& pose_msg);
  void PointCallback(const sensor_msgs::PointCloudConstPtr& point_msg);
  void ReloRelativePoseCallback(const nav_msgs::OdometryConstPtr& pose_msg);

 private:
  ros::NodeHandle nh_{"~"};
  PoseGraphConfig config_;
  PoseGraph pose_graph_ = PoseGraph();
  std::unique_ptr<CameraPoseVisualization> camera_pose_vis_;
  bool loop_closure_;
  camodocal::CameraPtr camera_;  // Note: internally it uses a shared pointer.
  bool visualise_imu_forward_;
  std::queue<sensor_msgs::ImageConstPtr> image_buffer_;
  std::queue<sensor_msgs::PointCloudConstPtr> point_buffer_;
  std::queue<nav_msgs::Odometry::ConstPtr> pose_buffer_;
  std::queue<Eigen::Vector3d> odometry_buffer_;
  std::mutex buffer_mutex_;
  std::mutex process_mutex_;
  Eigen::Vector3d last_t_ = {-100, -100, -100};
  std::string vins_result_path_;
  std::string image_topic_;
  nav_msgs::Path path_[10];
  nav_msgs::Path base_path_;
  nav_msgs::Path no_loop_path_;
  std::unique_ptr<CameraPoseVisualization> posegraph_visualization_;

  // ros parameters
  std::string config_file_path_;
  int visualization_shift_x_ = 0;
  int visualization_shift_y_ = 0;
  int skip_cnt_threshold_ = 0;
  int skip_cnt_ = 0;
  double skip_distance_ = 0.0;
  static const int skip_first_cnt_threshold = 10;
  const bool save_loop_path = true;
  int skip_first_cnt_ = 0;
  int frame_index_ = 0;
  int sequence_index_ = 1;
  double last_image_time_ = -1.0;  // TODO: consider using chrono timepoint
  PoseGraph::Pose imu_camera_pose_;
  std::thread measurement_thread_;
  std::thread keyboard_command_thread_;

  // ros subscribers
  ros::Subscriber sub_imu_forward_;
  ros::Subscriber sub_vio_;
  ros::Subscriber sub_image_;
  ros::Subscriber sub_pose_;
  ros::Subscriber sub_extrinsic_;
  ros::Subscriber sub_point_;
  ros::Subscriber sub_relo_relative_pose_;

  // ros publishers
  ros::Publisher pub_match_points_;
  ros::Publisher pub_camera_pose_visual_;
  ros::Publisher pub_key_odometrys_;
  ros::Publisher pub_vio_path_;
  ros::Publisher pub_pg_path_;
  ros::Publisher pub_base_path_;
  ros::Publisher pub_pose_graph_;
  ros::Publisher pub_path_[10];
  ros::Publisher pub_match_img_;
};
}  // namespace pose_graph

#endif /* POSE_GRAPH_ROS_POSE_GRAPH_NODE_HPP */
