#include "feature_tracker/feature_tracker.h"

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
void reduceVector(vector<T> &v, const vector<uchar> status) {
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
    : fisheye_(fisheye),
      run_histogram_equilisation_(run_histogram_equilisation),
      max_feature_count_per_image_(max_feature_count_per_image),
      min_distance_between_features_(min_distance_between_features),
      fundemental_matrix_ransac_threshold_(fundemental_matrix_threshold),
      fx_(fx),
      fy_(fy),
      feature_pruning_frequency_(feature_pruning_frequency),
      max_time_difference_(max_time_difference),
      previous_frame_time_(0.0),
      prev_prune_time_(0.0) {
  readIntrinsicParameter(camera_config_file);
  feature_pruning_period_ = 1.0 / feature_pruning_frequency;
  if (run_histogram_equilisation) {
    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));
  }
}

void FeatureTracker::RegisterEventObserver(
    std::shared_ptr<FeatureTrackerObserver> event_observer) {
  event_observer_ = event_observer;
  event_observer_->OnRegistered();
}

void FeatureTracker::ProcessNewFrame(cv::Mat new_frame,
                                     double current_image_time_s) {
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
  vector<cv::Point2f> current_points;
  vector<cv::Point2f> cur_un_pts;

  if (previous_points_.size() > 0) {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(previous_pre_processed_image_, pre_processed_img,
                             previous_points_, current_points, status, err,
                             cv::Size(21, 21), 3);

    for (size_t i = 0; i < current_points.size(); i++)
      if (status[i] && !inBorder(current_points[i], m_camera->imageWidth(),
                                 m_camera->imageHeight()))
        status[i] = 0;
    reduceVector(previous_points_, status);
    reduceVector(previous_undistorted_pts_, status);
    reduceVector(current_points, status);
    reduceVector(feature_ids_, status);
    reduceVector(feature_track_lengh_, status);

    for (size_t i = 0; i < current_points.size(); i++) {
      cur_un_pts.push_back(UndistortPoint(current_points[i], m_camera));
    }
    for (int &n : feature_track_lengh_) n++;
  }  // Prune and detect new points
  if (current_image_time_s > prev_prune_time_ + feature_pruning_period_) {
    PrunePointsUsingRansac(current_points, cur_un_pts, previous_points_,
                           previous_undistorted_pts_, feature_ids_,
                           feature_track_lengh_);
    DetectNewFeaturePoints(
        current_points, cur_un_pts, pre_processed_img,
        max_feature_count_per_image_ - current_points.size());

    vector<cv::Point2f> pts_velocity;
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
  DetectNewFeaturePoints(previous_points_, previous_undistorted_pts_,
                         pre_processed_img, max_feature_count_per_image_);

  previous_frame_time_ = current_time;
  prev_prune_time_ = current_time;
  previous_pre_processed_image_ = pre_processed_img;
  return;
}

cv::Mat FeatureTracker::setMask(vector<cv::Point2f> &curr_pts) {
  cv::Mat mask;
  if (fisheye_)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(m_camera->imageHeight(), m_camera->imageWidth(), CV_8UC1,
                   cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (size_t i = 0; i < curr_pts.size(); i++)
    cnt_pts_id.push_back(make_pair(feature_track_lengh_[i],
                                   make_pair(curr_pts[i], feature_ids_[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>> &a,
          const pair<int, pair<cv::Point2f, int>> &b) {
         return a.first > b.first;
       });

  curr_pts.clear();
  feature_ids_.clear();
  feature_track_lengh_.clear();

  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      curr_pts.push_back(it.second.first);
      feature_ids_.push_back(it.second.second);
      feature_track_lengh_.push_back(it.first);
      cv::circle(mask, it.second.first, min_distance_between_features_, 0, -1);
    }
  }
  return mask;
}

void FeatureTracker::AddPoints(
    vector<cv::Point2f> &curr_pts, vector<cv::Point2f> &cur_un_pts,
    const camodocal::CameraPtr m_camera,
    const vector<cv::Point2f> &newly_generated_points) {
  for (const cv::Point2f &p : newly_generated_points) {
    curr_pts.push_back(p);
    cur_un_pts.push_back(UndistortPoint(p, m_camera));
    feature_ids_.push_back(feature_counter_++);
    feature_track_lengh_.push_back(1);
  }
}

void FeatureTracker::DetectNewFeaturePoints(
    vector<cv::Point2f> &current_points,
    vector<cv::Point2f> &current_undistorted_points,
    const cv::Mat &pre_processed_img, int n_max_point_to_detect) {
  cv::Mat mask = setMask(current_points);
  if (n_max_point_to_detect > 0) {
    vector<cv::Point2f> newly_generated_points;
    cv::goodFeaturesToTrack(pre_processed_img, newly_generated_points,
                            n_max_point_to_detect, 0.01,
                            min_distance_between_features_, mask);
    AddPoints(current_points, current_undistorted_points, m_camera,
              newly_generated_points);
  }
}

vector<uchar> FeatureTracker::rejectWithF(
    const vector<cv::Point2f> &cur_un_pts,
    const vector<cv::Point2f> &prev_un_pts) const {
  vector<uchar> status;
  if (cur_un_pts.size() < 8) {
    return status;
  }

  vector<cv::Point2f> curr_scaled_undistorted_pts(cur_un_pts.size()),
      prev_scaled_undistorted_pts(prev_un_pts.size());

  for (size_t i = 0; i < prev_un_pts.size(); i++) {
    curr_scaled_undistorted_pts[i] =
        cv::Point2f(fx_ * cur_un_pts[i].x + m_camera->imageWidth() / 2.0,
                    fy_ * cur_un_pts[i].y + m_camera->imageHeight() / 2.0);

    prev_scaled_undistorted_pts[i] =
        cv::Point2f(fx_ * prev_un_pts[i].x + m_camera->imageWidth() / 2.0,
                    fy_ * prev_un_pts[i].y + m_camera->imageHeight() / 2.0);
  }

  cv::findFundamentalMat(prev_scaled_undistorted_pts,
                         curr_scaled_undistorted_pts, cv::FM_RANSAC,
                         fundemental_matrix_ransac_threshold_, 0.99, status);

  return status;
}

void FeatureTracker::PrunePointsUsingRansac(
    vector<cv::Point2f> &current_points,
    vector<cv::Point2f> &current_undistorted_points,
    vector<cv::Point2f> &previous_points,
    vector<cv::Point2f> &previous_undistorted_points, vector<int> &ids,
    vector<int> &track_counts) const {
  vector<uchar> status =
      rejectWithF(current_undistorted_points, previous_undistorted_points);
  reduceVector(current_points, status);
  reduceVector(current_undistorted_points, status);
  reduceVector(previous_points, status);
  reduceVector(previous_undistorted_points, status);
  reduceVector(ids, status);
  reduceVector(track_counts, status);
}
void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
  spdlog::info("reading paramerter of camera {}", calib_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::GetPointVelocty(
    double dt, const vector<cv::Point2f> &cur_un_pts,
    const vector<cv::Point2f> &prev_un_pts,
    vector<cv::Point2f> &pts_velocity_out) const {
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
  m_camera->liftProjective(a, b);
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