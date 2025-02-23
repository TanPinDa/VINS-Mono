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
}

void FeatureTracker::RegisterEventObserver(
    std::shared_ptr<FeatureTrackerObserver> event_observer) {
  event_observer_ = event_observer;
  std::cout << "test" << std::endl;
  event_observer_->OnRegistered();
  std::cout << "test2" << std::endl;
}

void FeatureTracker::ProcessNewFrame(cv::Mat new_frame,
                                     double current_image_time_s) {
  std::cout << "Got a frame" << std::endl;
  if (is_first_frame_) {
    std::cout << "First frame" << std::endl;
    is_first_frame_ = false;
    first_frame_time_ = current_image_time_s;
    previous_frame_time_ = current_image_time_s;
    prev_prune_time = current_image_time_s;
    return;
  }

  if (current_image_time_s > previous_frame_time_ + max_time_difference_) {
    std::cout << "Time diff too large" << std::endl;
    RestartTracker();
    if (event_observer_) {
      event_observer_->OnDurationBetweenFrameTooLarge(current_image_time_s,
                                                      previous_frame_time_);
    }
    return;
  }

  if (current_image_time_s < previous_frame_time_) {
    std::cout << "went back in time" << std::endl;
    RestartTracker();
    if (event_observer_) {
      event_observer_->OnDurationBetweenFrameTooLarge(current_image_time_s,
                                                      previous_frame_time_);
    }
    return;
  }

  std::cout << "reading image" << std::endl;
  readImage(new_frame, current_image_time_s);

  for (unsigned int i = 0;; i++) {
    bool completed = false;
    // std::cout << "updating id" << std::endl;
    completed |= updateID(i);
    if (!completed) break;
  }
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

void FeatureTracker::setMask() {
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

void FeatureTracker::addPoints() {
  for (auto &p : n_pts) {
    curr_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time) {
  cv::Mat img;
  TicToc t_r;
  cur_time = _cur_time;
  std::cout << "test1" << std::endl;
  if (run_histogram_equilisation_) {
    std::cout << "Doing histogram equailsation" << std::endl;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    TicToc t_c;
    clahe->apply(_img, img);
    std::cout << "Applied histogram equailsation" << std::endl;
    spdlog::debug("CLAHE costs: {}ms", t_c.toc());
  } else
    img = _img;

  if (cur_img_.empty()) {
    std::cout << "First iamge to setting data" << std::endl;
    prev_img_ = cur_img_ = img;
  } else {
    std::cout << "set curr img" << std::endl;
    cur_img_ = img;
  }

  curr_pts.clear();

  if (prev_pts.size() > 0) {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_img_, cur_img_, prev_pts, curr_pts, status,
                             err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(curr_pts.size()); i++)
      if (status[i] && !inBorder(curr_pts[i], m_camera->imageWidth(),
                                 m_camera->imageHeight()))
        status[i] = 0;
    reduceVector(prev_pts, status);
    reduceVector(curr_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    spdlog::debug("temporal optical flow costs: {}ms", t_o.toc());
  }

  for (auto &n : track_cnt) n++;

  // Prune and detect new points
  bool is_prune_and_detect_new_points =
      _cur_time > prev_prune_time + feature_pruning_period_;
  if (is_prune_and_detect_new_points) {
    std::cout << "Time to prune and find new" << std::endl;
    rejectWithF();

    spdlog::debug("set mask begins");
    TicToc t_m;
    setMask();
    spdlog::debug("set mask costs {}ms", t_m.toc());

    spdlog::debug("detect feature begins");
    TicToc t_t;
    int n_max_cnt =
        max_feature_count_per_image_ - static_cast<int>(curr_pts.size());
    if (n_max_cnt > 0) {
      if (mask.empty()) cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1) cout << "mask type wrong " << endl;
      if (mask.size() != cur_img_.size()) cout << "wrong size " << endl;
      cv::goodFeaturesToTrack(cur_img_, n_pts, n_max_cnt, 0.01,
                              min_distance_between_features_, mask);
    } else
      n_pts.clear();
    spdlog::debug("detect feature costs: {}ms", t_t.toc());

    spdlog::debug("add feature begins");
    TicToc t_a;
    addPoints();
    spdlog::debug("selectFeature costs: {}ms", t_a.toc());
    prev_prune_time = _cur_time;

    // event_observer_->OnProcessedImage(_img,_cur_time,);
  }

  undistortedPoints();
  if (is_prune_and_detect_new_points) {
    if (event_observer_) {
      event_observer_->OnProcessedImage(_img, _cur_time, curr_pts, cur_un_pts,
                                        ids, track_cnt, pts_velocity);
    }
  }
  prev_img_ = cur_img_;
  prev_pts = curr_pts;
  prev_time = cur_time;
}

void FeatureTracker::rejectWithF() {
  if (curr_pts.size() >= 8) {
    spdlog::debug("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(prev_pts.size()),
        un_forw_pts(curr_pts.size());
    for (unsigned int i = 0; i < prev_pts.size(); i++) {
      Eigen::Vector3d tmp_p;
      m_camera->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y),
                               tmp_p);
      tmp_p.x() = fx_ * tmp_p.x() / tmp_p.z() + m_camera->imageWidth() / 2.0;
      tmp_p.y() = fy_ * tmp_p.y() / tmp_p.z() + m_camera->imageHeight() / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera->liftProjective(Eigen::Vector2d(curr_pts[i].x, curr_pts[i].y),
                               tmp_p);
      tmp_p.x() = fx_ * tmp_p.x() / tmp_p.z() + m_camera->imageWidth() / 2.0;
      tmp_p.y() = fy_ * tmp_p.y() / tmp_p.z() + m_camera->imageHeight() / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC,
                           fundemental_matrix_ransac_threshold_, 0.99, status);
    int size_a = prev_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(curr_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    spdlog::debug("FM ransac: {0} -> {1}: {2}", size_a, curr_pts.size(),
                  1.0 * curr_pts.size() / size_a);
    spdlog::debug("FM ransac costs: {}ms", t_f.toc());
  }
}

bool FeatureTracker::updateID(unsigned int i) {
  if (i < ids.size()) {
    if (ids[i] == -1) ids[i] = n_id++;
    return true;
  } else
    return false;
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

void FeatureTracker::undistortedPoints() {
  map<int, cv::Point2f> cur_un_pts_map;
  cur_un_pts.clear();

  for (unsigned int i = 0; i < curr_pts.size(); i++) {
    Eigen::Vector2d a(curr_pts[i].x, curr_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(
        make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    // printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
  }
  // caculate points velocity
  if (!prev_un_pts_map.empty()) {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++) {
      if (ids[i] != -1) {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end()) {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        } else
          pts_velocity.push_back(cv::Point2f(0, 0));
      } else {
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  } else {
    for (unsigned int i = 0; i < prev_pts.size(); i++) {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}
