#include "feature_tracker/feature_tracker.h"

#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "spdlog/spdlog.h"
int FeatureTracker::feature_counter_ = 0;

bool inBorder(const cv::Point2f &pt, const uint col, const uint row,
              const uint border_size = 1) {
  int x = cvRound(pt.x), y = cvRound(pt.y);
  return (x >= border_size && x < col - border_size) &&
         (y >= border_size && y < row - border_size);
}

template <typename T>
void reduceVector(std::vector<T> &v, const std::vector<uchar> status) {
  int j = 0;
  for (size_t i = 0; i < int(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

FeatureTracker::FeatureTracker(std::string camera_config_file, bool fisheye,
                               bool run_histogram_equilisation,
                               uint max_feature_count_per_image,
                               uint min_distance_between_features,
                               double fundemental_matrix_threshold, double fx,
                               double fy, double feature_pruning_frequency,
                               double max_time_difference)
    : run_histogram_equilisation_(run_histogram_equilisation),
      max_feature_count_per_image_(max_feature_count_per_image),
      min_distance_between_features_(min_distance_between_features),
      fundemental_matrix_ransac_threshold_(fundemental_matrix_threshold),
      fx_(fx),
      fy_(fy),
      feature_pruning_frequency_(feature_pruning_frequency),
      max_time_difference_(max_time_difference),
      previous_frame_time_(0.0),
      prev_prune_time_(0.0) {
  camera_model_ =
      CameraFactory::instance()->generateCameraFromYamlFile(camera_config_file);
  feature_pruning_period_ = 1.0 / feature_pruning_frequency;
  if (run_histogram_equilisation) {
    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
  }

  if (fisheye) std::cout << "!!!FISH EYE NOT WORKING";
  // base_mask_ = fisheye_mask.clone();
  else
    base_mask_ = cv::Mat(camera_model_->imageHeight(),
                         camera_model_->imageWidth(), CV_8UC1, cv::Scalar(255));
}

void FeatureTracker::RegisterEventObserver(
    std::shared_ptr<FeatureTrackerObserver> event_observer) {
  event_observer_ = event_observer;
  event_observer_->OnRegistered();
}

void FeatureTracker::ProcessNewFrame(const cv::Mat &new_frame,
                                     const double current_image_time_s) {
  cv::Mat pre_processed_img;
  if (run_histogram_equilisation_) {
    clahe_->apply(new_frame, pre_processed_img);
  } else
    pre_processed_img = new_frame;

  if (current_image_time_s > previous_frame_time_ + max_time_difference_) {
    if (event_observer_) {
      event_observer_->OnDurationBetweenFrameTooLarge(current_image_time_s,
                                                      previous_frame_time_);
    }
    RestartTracker(pre_processed_img, current_image_time_s);
    return;
  }

  if (current_image_time_s < previous_frame_time_) {
    if (event_observer_) {
      event_observer_->OnDurationBetweenFrameTooLarge(current_image_time_s,
                                                      previous_frame_time_);
    }
    RestartTracker(pre_processed_img, current_image_time_s);
    return;
  }
  // Find new feature points by optical flow
  std::vector<cv::Point2f> current_points;
  std::vector<cv::Point2f> cur_un_pts;

  if (previous_points_.size() > 0) {
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(previous_pre_processed_image_, pre_processed_img,
                             previous_points_, current_points, status, err,
                             cv::Size(21, 21), 3);

    for (size_t i = 0; i < current_points.size(); i++) {
      cur_un_pts.push_back(UndistortPoint(current_points[i], camera_model_));
    }

    for (size_t i = 0; i < current_points.size(); i++)
      if (status[i] && !inBorder(current_points[i], camera_model_->imageWidth(),
                                 camera_model_->imageHeight()))
        status[i] = 0;

    PrunePoints(current_points, cur_un_pts, previous_points_,
                previous_undistorted_pts_, feature_track_lengh_, feature_ids_,
                status);

    for (int &n : feature_track_lengh_) n++;
  }

  // Prune and detect new points
  if (current_image_time_s > prev_prune_time_ + feature_pruning_period_) {
    if (current_points.size() > 8) {
      std::vector<uchar> ransac_status;

      RejectUsingRansac(cur_un_pts, previous_undistorted_pts_, ransac_status);
      PrunePoints(current_points, cur_un_pts, previous_points_,
                  previous_undistorted_pts_, feature_track_lengh_, feature_ids_,
                  ransac_status);
    }

    std::vector<uchar> status;
    cv::Mat mask = CreateMask(current_points, feature_track_lengh_, status);
    PrunePoints(current_points, cur_un_pts, previous_points_,
                previous_undistorted_pts_, feature_track_lengh_, feature_ids_,
                status);

    int n_max_point_to_detect =
        max_feature_count_per_image_ - current_points.size();
    if (n_max_point_to_detect > 0) {
      AddPoints(pre_processed_img, mask, n_max_point_to_detect,
                min_distance_between_features_, camera_model_, current_points,
                cur_un_pts, feature_track_lengh_, feature_ids_);
    }

    std::vector<cv::Point2f> pts_velocity;
    GetPointVelocty(current_image_time_s - previous_frame_time_, cur_un_pts,
                    previous_undistorted_pts_, pts_velocity);

    if (event_observer_) {
      event_observer_->OnProcessedImage(
          pre_processed_img, current_image_time_s, current_points, cur_un_pts,
          feature_ids_, feature_track_lengh_, pts_velocity);
    }
    prev_prune_time_ = current_image_time_s;
  }

  previous_undistorted_pts_ = cur_un_pts;
  previous_pre_processed_image_ = pre_processed_img;
  previous_points_ = current_points;
  previous_frame_time_ = current_image_time_s;
}

void FeatureTracker::RestartTracker(const cv::Mat &pre_processed_img,
                                    double current_time) {
  if (event_observer_) {
    event_observer_->OnRestart();
  }
  previous_points_.clear();
  previous_undistorted_pts_.clear();
  feature_ids_.clear();
  feature_track_lengh_.clear();

  std::vector<cv::Point2f> newly_generated_points;
  AddPoints(pre_processed_img, base_mask_, max_feature_count_per_image_,
            min_distance_between_features_, camera_model_, previous_points_,
            previous_undistorted_pts_, feature_track_lengh_, feature_ids_);

  previous_frame_time_ = current_time;
  prev_prune_time_ = current_time;
  previous_pre_processed_image_ = pre_processed_img;
  return;
}

void FeatureTracker::AddPoints(
    const cv::Mat image, const cv::Mat mask, const int max_number_new_of_points,
    const int min_distance_between_points, const camodocal::CameraPtr m_camera,
    std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &undistorted_points,
    std::vector<int> &track_length, std::vector<int> &feature_ids) {
  std::vector<cv::Point2f> newly_generated_points;
  cv::goodFeaturesToTrack(image, newly_generated_points,
                          max_number_new_of_points, 0.01,
                          min_distance_between_points, mask);
  for (const cv::Point2f &p : newly_generated_points) {
    points.push_back(p);
    undistorted_points.push_back(UndistortPoint(p, m_camera));
    feature_ids.push_back(feature_counter_++);
    track_length.push_back(1);
  }
}

void FeatureTracker::PrunePoints(std::vector<cv::Point2f> &curr_points,
                                 std::vector<cv::Point2f> &curr_un_points,
                                 std::vector<cv::Point2f> &prev_points,
                                 std::vector<cv::Point2f> &prev_un_points,
                                 std::vector<int> &ids, std::vector<int> &track_counts,
                                 const std::vector<uchar> &status) {
  reduceVector(curr_points, status);
  reduceVector(curr_un_points, status);
  reduceVector(prev_points, status);
  reduceVector(prev_un_points, status);
  reduceVector(ids, status);
  reduceVector(track_counts, status);
}

cv::Mat FeatureTracker::CreateMask(std::vector<cv::Point2f> &curr_pts,
                                   std::vector<int> &track_length,
                                   std::vector<uchar> &status_out) {
  cv::Mat mask = base_mask_.clone();

  std::vector<size_t> indices(track_length.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [track_length](int A, int B) -> bool {
              return track_length[A] < track_length[B];
            });

  status_out.assign(curr_pts.size(), 0);

  for (size_t i : indices) {
    if (mask.at<uchar>(curr_pts[i]) == 255) {
      cv::circle(mask, curr_pts[i], min_distance_between_features_, 0, -1);
      status_out[i] = 1;
    }
  }

  return mask;
}

void FeatureTracker::RejectUsingRansac(
    const std::vector<cv::Point2f> &current_undistorted_points,
    const std::vector<cv::Point2f> &previous_undistorted_points,
    std::vector<uchar> &status_out) const {
  status_out.clear();
  size_t n_points = current_undistorted_points.size();

  if (n_points > 7) {
    std::vector<cv::Point2f> curr_scaled_undistorted_pts(n_points),
        prev_scaled_undistorted_pts(n_points);

    for (size_t i = 0; i < n_points; i++) {
      curr_scaled_undistorted_pts[i] =
          cv::Point2f(fx_ * current_undistorted_points[i].x +
                          camera_model_->imageWidth() / 2.0,
                      fy_ * current_undistorted_points[i].y +
                          camera_model_->imageHeight() / 2.0);

      prev_scaled_undistorted_pts[i] =
          cv::Point2f(fx_ * previous_undistorted_points[i].x +
                          camera_model_->imageWidth() / 2.0,
                      fy_ * previous_undistorted_points[i].y +
                          camera_model_->imageHeight() / 2.0);
    }

    cv::findFundamentalMat(
        prev_scaled_undistorted_pts, curr_scaled_undistorted_pts, cv::FM_RANSAC,
        fundemental_matrix_ransac_threshold_, 0.99, status_out);
  }
}

void FeatureTracker::GetPointVelocty(
    double dt, const std::vector<cv::Point2f> &cur_un_pts,
    const std::vector<cv::Point2f> &prev_un_pts,
    std::vector<cv::Point2f> &pts_velocity_out) const {
  pts_velocity_out.clear();

  for (size_t i = 0; i < cur_un_pts.size(); i++) {
    if (i < prev_un_pts.size()) {
      cv::Point2f point_velocity((cur_un_pts[i].x - prev_un_pts[i].x) / dt,
                                 (cur_un_pts[i].y - prev_un_pts[i].y) / dt);
      pts_velocity_out.push_back(point_velocity);
    } else {
      pts_velocity_out.push_back(cv::Point2f(0, 0));
    }
  }
}

cv::Point2f FeatureTracker::UndistortPoint(
    const cv::Point2f point, const camodocal::CameraPtr camera) const {
  Eigen::Vector2d a(point.x, point.y);
  Eigen::Vector3d b;
  camera->liftProjective(a, b);
  return cv::Point2f(b.x() / b.z(), b.y() / b.z());
}

std::string FeatureTracker::GenerateStateString() const {
  std::stringstream ss;
  ss << "REPORTING CURRENT STATE"
     << "\n\t Previous time is:" << previous_frame_time_
     << "\n\t Previous prune time is:" << prev_prune_time_
     << "\n\t Previous number of points is: " << previous_points_.size()
     << "\n\t Previous number of undistorted points is: "
     << previous_undistorted_pts_.size();
  return ss.str();
}