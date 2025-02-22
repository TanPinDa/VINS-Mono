/**
 * @file pose_graph_event_observer.hpp
 * @brief
 * @date 20-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#ifndef DETAILS_POSE_GRAPH_EVENT_OBSERVER_HPP
#define DETAILS_POSE_GRAPH_EVENT_OBSERVER_HPP

#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "pose_graph/details/keyframe.h"

namespace pose_graph {
class PoseGraphEventObserver
    : public std::enable_shared_from_this<PoseGraphEventObserver> {
 public:
  PoseGraphEventObserver() = default;
  virtual ~PoseGraphEventObserver() = default;

  virtual void OnPoseGraphLoaded() = 0;
  virtual void OnPoseGraphSaved() = 0;
  virtual void OnKeyFrameAdded(KeyFrame::Attributes kf_attribute) = 0;
  virtual void OnKeyFrameLoaded(KeyFrame::Attributes kf_attribute,
                                int count) = 0;
  virtual void OnKeyFrameConnectionFound(
      KeyFrame::Attributes current_kf_attribute,
      KeyFrame::Attributes old_kf_attribute,
      std::vector<cv::Point2f> matched_2d_old_norm,
      std::vector<double> matched_id, cv::Mat& thumb_image) = 0;
  virtual void OnPoseGraphOptimization(
      std::vector<KeyFrame::Attributes> kf_attributes) = 0;
  virtual void OnNewSequentialEdge(Eigen::Vector3d p1, Eigen::Vector3d p2) = 0;
  virtual void OnNewLoopEdge(Eigen::Vector3d p1, Eigen::Vector3d p2) = 0;
};
}  // namespace pose_graph

#endif /* DETAILS_POSE_GRAPH_EVENT_OBSERVER_HPP */
