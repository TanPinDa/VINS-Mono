#include "feature_tracker/feature_tracker.h"

#include <opencv2/opencv.hpp>

#include "spdlog/spdlog.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt, uint col, uint row) {
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE &&
         BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
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
      max_time_difference_(max_time_difference) {
  readIntrinsicParameter(camera_config_file);
  feature_pruning_period_ = 1.0 / feature_pruning_frequency;
  if (run_histogram_equilisation) {
    clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
  }
}

void FeatureTracker::RegisterEventObserver(
    std::shared_ptr<FeatureTrackerObserver> event_observer) {
  event_observer_ = event_observer;
  event_observer_->OnRegistered();
}

void FeatureTracker::ProcessNewFrame(cv::Mat new_frame,
                                     double current_image_time_s) {
  if (is_first_frame_) {
    is_first_frame_ = false;
    first_frame_time_ = current_image_time_s;
    previous_frame_time_ = current_image_time_s;
    prev_prune_time = current_image_time_s;
    return;
  }

  if (current_image_time_s > previous_frame_time_ + max_time_difference_) {
    RestartTracker();
    if (event_observer_) {
      event_observer_->OnDurationBetweenFrameTooLarge(current_image_time_s,
                                                      previous_frame_time_);
    }
    return;
  }

  if (current_image_time_s < previous_frame_time_) {
    RestartTracker();
    if (event_observer_) {
      event_observer_->OnDurationBetweenFrameTooLarge(current_image_time_s,
                                                      previous_frame_time_);
    }
    return;
  }
  readImage(new_frame, current_image_time_s);

}

void FeatureTracker::RestartTracker() {
  is_first_frame_ = true;
  first_frame_time_ = 0;
  previous_frame_time_ = 0;

  if (event_observer_) {
    event_observer_->OnRestart();
  }
  return;
}

void FeatureTracker::setMask(vector<cv::Point2f> &curr_pts) {
  if (fisheye_)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(m_camera->imageHeight(), m_camera->imageWidth(), CV_8UC1,
                   cv::Scalar(255));

  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < curr_pts.size(); i++)
    cnt_pts_id.push_back(
        make_pair(track_cnt[i], make_pair(curr_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>> &a,
          const pair<int, pair<cv::Point2f, int>> &b) {
         return a.first > b.first;
       });

  curr_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      curr_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, min_distance_between_features_, 0, -1);
    }
  }
}

void FeatureTracker::addPoints(
    vector<cv::Point2f> &curr_pts, vector<cv::Point2f> &cur_un_pts,
    const camodocal::CameraPtr m_camera,
    const vector<cv::Point2f> &newly_generated_points) {
  for (auto &p : newly_generated_points) {
    int new_point_id = ++n_id;
    curr_pts.push_back(p);
    cur_un_pts.push_back(UndistortPoint(p, m_camera));
    ids.push_back(new_point_id);
    track_cnt.push_back(1);
  }
}

void FeatureTracker::readImage(const cv::Mat &img, double current_time) {
  cv::Mat pre_processed_img;
  vector<cv::Point2f> current_points;

  if (run_histogram_equilisation_) {
    clahe->apply(img, pre_processed_img);
  } else
    pre_processed_img = img;

  if (prev_img_.empty()) {
    prev_img_ = pre_processed_img;
  }

  if (prev_pts.size() > 0) {
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img_, pre_processed_img, prev_pts,
                             current_points, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(current_points.size()); i++)
      if (status[i] && !inBorder(current_points[i], m_camera->imageWidth(),
                                 m_camera->imageHeight()))
        status[i] = 0;
    reduceVector(prev_pts, status);
    reduceVector(prev_un_pts, status);
    reduceVector(current_points, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
  }
  for (auto &n : track_cnt) n++;

  vector<cv::Point2f> cur_un_pts;

  for (unsigned int i = 0; i < current_points.size(); i++) {
    cur_un_pts.push_back(UndistortPoint(current_points[i], m_camera));

    // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
  }
  // Prune and detect new points
  bool is_prune_and_detect_new_points =
      current_time > prev_prune_time + feature_pruning_period_;

  if (is_prune_and_detect_new_points) {
    vector<uchar> status = rejectWithF(cur_un_pts, prev_un_pts);
    int size_a = prev_un_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(prev_un_pts, status);
    reduceVector(current_points, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);

    spdlog::debug("FM ransac: {0} -> {1}: {2}", size_a, current_points.size(),
                  1.0 * current_points.size() / size_a);

    spdlog::debug("set mask begins");

    setMask(current_points);

    spdlog::debug("detect feature begins");
    int n_max_cnt =
        max_feature_count_per_image_ - static_cast<int>(current_points.size());
    if (n_max_cnt > 0) {
      if (mask.empty()) cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1) cout << "mask type wrong " << endl;
      if (mask.size() != pre_processed_img.size())
        cout << "wrong size " << endl;

      vector<cv::Point2f> newly_generated_points;
      cv::goodFeaturesToTrack(pre_processed_img, newly_generated_points,
                              n_max_cnt, 0.01, min_distance_between_features_,
                              mask);
      addPoints(current_points, cur_un_pts, m_camera, newly_generated_points);
    }

    vector<cv::Point2f> pts_velocity;

    GetPointVelocty(current_time - prev_time, cur_un_pts, prev_un_pts,
                    pts_velocity);

    if (event_observer_) {
      event_observer_->OnProcessedImage(img, current_time, current_points,
                                        cur_un_pts, ids, track_cnt,
                                        pts_velocity);
    }
    prev_prune_time = current_time;
  }

  prev_un_pts = cur_un_pts;
  prev_img_ = pre_processed_img;
  prev_pts = current_points;
  prev_time = current_time;
}
vector<uchar> FeatureTracker::rejectWithF(
    const vector<cv::Point2f> &cur_un_pts,
    const vector<cv::Point2f> &prev_un_pts) {
  vector<uchar> status;
  if (cur_un_pts.size() < 8) {
    return status;
  }

  spdlog::debug("FM ransac begins");
  vector<cv::Point2f> curr_scaled_undistorted_pts(cur_un_pts.size()),
      prev_scaled_undistorted_pts(prev_un_pts.size());

  for (unsigned int i = 0; i < prev_un_pts.size(); i++) {
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



void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
  spdlog::info("reading paramerter of camera {}", calib_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
  cv::Mat undistortedImg(m_camera->imageHeight() + 600,
                         m_camera->imageWidth() + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < m_camera->imageWidth(); i++)
    for (int j = 0; j < m_camera->imageHeight(); j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      // printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++) {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * fx_ + m_camera->imageWidth() / 2;
    pp.at<float>(1, 0) =
        undistortedp[i].y() * fy_ + m_camera->imageHeight() / 2;
    pp.at<float>(2, 0) = 1.0;
    // cout << trackerData[0].K << endl;
    // printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    // printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 &&
        pp.at<float>(1, 0) + 300 < m_camera->imageHeight() + 600 &&
        pp.at<float>(0, 0) + 300 >= 0 &&
        pp.at<float>(0, 0) + 300 < m_camera->imageWidth() + 600) {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300,
                               pp.at<float>(0, 0) + 300) =
          prev_img_.at<uchar>(distortedp[i].y(), distortedp[i].x());
    } else {
      // spdlog::error("({0} {1}) -> ({2} {3})", distortedp[i].y,
      // distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

void FeatureTracker::GetPointVelocty(double dt,
                                     const vector<cv::Point2f> &cur_un_pts,
                                     const vector<cv::Point2f> &prev_un_pts,
                                     vector<cv::Point2f> &pts_velocity_out) {
  pts_velocity_out.clear();

  for (unsigned int i = 0; i < prev_un_pts.size(); i++) {
    cv::Point2f point_velocity((cur_un_pts[i].x - prev_un_pts[i].x) / dt,
                               (cur_un_pts[i].y - prev_un_pts[i].y) / dt);
    pts_velocity_out.push_back(point_velocity);
  }
}

cv::Point2f FeatureTracker::UndistortPoint(
    const cv::Point2f point, const camodocal::CameraPtr camera) const {
  Eigen::Vector2d a(point.x, point.y);
  Eigen::Vector3d b;
  m_camera->liftProjective(a, b);
  return cv::Point2f(b.x() / b.z(), b.y() / b.z());
}
