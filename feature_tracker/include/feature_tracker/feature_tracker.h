#pragma once

#include <execinfo.h>

#include <csignal>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "feature_tracker/feature_tracker_observer.hpp"
#include "feature_tracker/parameters.h"
#include "feature_tracker/tic_toc.h"

using namespace camodocal;
using namespace Eigen;

class FeatureTracker {
 public:
  FeatureTracker(std::string camera_config_file, bool fisheye,
                 bool run_histogram_equilisation,
                 uint max_feature_count_per_image,
                 uint min_distance_between_features,
                 double fundemental_matrix_ransac_threshold, double fx,
                 double fy, double feature_pruning_frequency,
                 double max_time_difference);
  void RegisterEventObserver(
      std::shared_ptr<FeatureTrackerObserver> event_observer);
  void ProcessNewFrame(const cv::Mat &img, const double time_s);

 private:
  void RestartTracker(const cv::Mat &pre_processed_img, double current_time);

  void AddPoints(const cv::Mat image, const cv::Mat mask,
                 const int max_number_new_of_points,
                 const int min_distance_between_points,
                 const camodocal::CameraPtr m_camera,
                 std::vector<cv::Point2f> &points,
                 std::vector<cv::Point2f> &undistorted_points,
                 std::vector<int> &track_length, std::vector<int> &feature_ids);

  void PrunePoints(std::vector<cv::Point2f> &curr_points,
                   std::vector<cv::Point2f> &curr_un_points,
                   std::vector<cv::Point2f> &prev_points,
                   std::vector<cv::Point2f> &prev_un_points,
                   std::vector<int> &ids, std::vector<int> &track_counts,
                   const std::vector<uchar> &status);

  cv::Mat CreateMask(std::vector<cv::Point2f> &curr_pts,
                     std::vector<int> &track_length,
                     std::vector<uchar> &status_out);

  void RejectUsingRansac(const std::vector<cv::Point2f> &cur_un_pts,
                         const std::vector<cv::Point2f> &prev_un_pts,
                         std::vector<uchar> &status_out) const;

  void GetPointVelocty(double dt, const std::vector<cv::Point2f> &cur_un_pts,
                       const std::vector<cv::Point2f> &prev_un_pts,
                       std::vector<cv::Point2f> &pts_velocity_out) const;

  cv::Point2f UndistortPoint(const cv::Point2f point,
                             const camodocal::CameraPtr camera) const;

  std::string GenerateStateString() const;

  cv::Mat base_mask_;
  cv::Mat previous_pre_processed_image_;
  double previous_frame_time_;

  std::vector<cv::Point2f> previous_undistorted_pts_;
  std::vector<cv::Point2f> previous_points_;
  std::vector<int> feature_ids_;
  std::vector<int> feature_track_lengh_;
  double prev_prune_time_;

  double fx_;
  double fy_;
  camodocal::CameraPtr camera_model_;

  static int feature_counter_;  // Static to ensure unique id between different
                                // instances

  cv::Ptr<cv::CLAHE> clahe_;

  bool fisheye_;
  bool run_histogram_equilisation_;
  uint max_feature_count_per_image_;
  uint min_distance_between_features_;
  double fundemental_matrix_ransac_threshold_;

  double max_time_difference_;
  double feature_pruning_frequency_;
  double feature_pruning_period_;
  std::shared_ptr<FeatureTrackerObserver> event_observer_;
};
