/**
 * @file pose_graph_event_observer_impl.cpp
 * @brief
 * @date 09-03-2025
 *
 * @copyright Copyright (c) 2025 Cheo Kee Jin.
 */

#ifndef SRC_POSE_GRAPH_EVENT_OBSERVER_IMPL_HPP
#define SRC_POSE_GRAPH_EVENT_OBSERVER_IMPL_HPP

#include <memory>

#include <opencv2/opencv.hpp>

#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include "pose_graph/details/pose_graph_event_observer.hpp"

#include "pose_graph_ros/pose_graph_node.hpp"
#include "pose_graph_ros/utility/CameraPoseVisualization.h"

namespace pose_graph {
class PoseGraphNode::PoseGraphEventObserverImpl
    : public PoseGraphEventObserver {
 public:
  PoseGraphEventObserverImpl(ros::NodeHandle& nh, std::string config_file_path);
  ~PoseGraphEventObserverImpl() = default;

  void Publish(int sequence_count);
  void ResetPoseGraphVisualisation();

 private:
  bool ReadParameters();
  void StartPublishers();

  // PoseGraphEventObserver callbacks
  void OnPoseGraphLoaded() final;
  void OnPoseGraphSaved() final;
  void OnKeyFrameAdded(KeyFrame::Attributes kf_attribute,
                       int sequence_count) final;
  void OnKeyFrameLoaded(KeyFrame::Attributes kf_attribute, int count,
                        int sequence_count) final;
  void OnKeyFrameConnectionFound(KeyFrame::Attributes current_kf_attribute,
                                 KeyFrame::Attributes old_kf_attribute,
                                 std::vector<cv::Point2f> matched_2d_old_norm,
                                 std::vector<double> matched_id,
                                 cv::Mat& thumb_image) final;
  void OnPoseGraphOptimization(std::vector<KeyFrame::Attributes> kf_attributes,
                               int sequence_count) final;
  void OnNewSequentialEdge(Vector3d p1, Vector3d p2) final;
  void OnNewLoopEdge(Vector3d p1, Vector3d p2) final;

 private:
  ros::NodeHandle& nh_;
  nav_msgs::Path path_[10];
  nav_msgs::Path base_path_;
  std::string vins_result_path_;
  bool fast_relocalization_{false};
  const bool save_loop_path = true;
  int visualization_shift_x_ = 0;
  int visualization_shift_y_ = 0;

  // ros publishers
  ros::Publisher pub_match_points_;
  ros::Publisher pub_pg_path_;
  ros::Publisher pub_base_path_;
  ros::Publisher pub_pose_graph_;
  ros::Publisher pub_path_[10];
  ros::Publisher pub_match_img_;
  std::unique_ptr<CameraPoseVisualization> posegraph_visualization_;
};
}  // namespace pose_graph

#endif /* SRC_POSE_GRAPH_EVENT_OBSERVER_IMPL_HPP */
