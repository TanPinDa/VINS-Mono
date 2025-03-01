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

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

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
  void ProcessNewFrame(cv::Mat img, double time_s);

 private:
  void readImage(const cv::Mat &pre_processed_img, double current_time);
  void RestartTracker(const cv::Mat &pre_processed_img, double current_time);
  cv::Mat setMask(vector<cv::Point2f> &curr_pts);

  void addPoints(vector<cv::Point2f> &curr_pts, vector<cv::Point2f> &cur_un_pts,
                 const camodocal::CameraPtr m_camera,
                 const vector<cv::Point2f> &newly_generated_points);

  cv::Point2f UndistortPoint(const cv::Point2f point,
                             const camodocal::CameraPtr camera) const;
  void PrunePointsUsingRansac(vector<cv::Point2f> &curr_points,
                              vector<cv::Point2f> &curr_un_points,
                              vector<cv::Point2f> &prev_points,
                              vector<cv::Point2f> &prev_un_points,
                              vector<int> &ids, vector<int> &track_counts);

  void DetectNewFeaturePoints(vector<cv::Point2f> &current_points,
                              vector<cv::Point2f> &current_undistorted_points,
                              const cv::Mat &pre_processed_img,
                              int n_max_point_to_detect);
  void readIntrinsicParameter(const string &calib_file);

  void showUndistortion(const string &name);

  vector<uchar> rejectWithF(const vector<cv::Point2f> &cur_un_pts,
                            const vector<cv::Point2f> &prev_un_pts);

  void GetPointVelocty(double dt, const vector<cv::Point2f> &cur_un_pts,
                       const vector<cv::Point2f> &prev_un_pts,
                       vector<cv::Point2f> &pts_velocity_out);

  std::string GenerateStateString();

  cv::Mat fisheye_mask;

  cv::Mat prev_img_;
  double previous_frame_time_;
  vector<cv::Point2f> prev_un_pts;
  vector<cv::Point2f> prev_pts;
  vector<int> feature_ids_;
  vector<int> track_cnt;
  double prev_prune_time_;

  double fx_;
  double fy_;
  camodocal::CameraPtr m_camera;

  static int n_id;

  cv::Ptr<cv::CLAHE> clahe;

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
