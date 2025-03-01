#include "pose_graph/details/keyframe.h"

#include "DBoW/DBoW2.h"
#include "DVision.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status) {
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i]) v[j++] = v[i];
  v.resize(j);
}

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i,
                   Matrix3d &_vio_R_w_i, cv::Mat &_image,
                   vector<cv::Point3f> &_point_3d,
                   vector<cv::Point2f> &_point_2d_uv,
                   vector<cv::Point2f> &_point_2d_norm,
                   vector<double> &_point_id, int _sequence, int image_height,
                   int image_width, std::string brief_pattern_file_path,
                   bool debug_image, camodocal::CameraPtr camera)
    : image_height_(image_height),
      image_width_(image_width),
      brief_pattern_file_path_(brief_pattern_file_path),
      debug_image_(debug_image) {
  time_stamp = _time_stamp;
  index = _index;
  vio_T_w_i = _vio_T_w_i;
  vio_R_w_i = _vio_R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
  origin_vio_T = vio_T_w_i;
  origin_vio_R = vio_R_w_i;
  image = _image.clone();
  cv::resize(image, thumbnail, cv::Size(80, 60));
  point_3d = _point_3d;
  point_2d_uv = _point_2d_uv;
  point_2d_norm = _point_2d_norm;
  point_id = _point_id;
  has_loop = false;
  loop_index = -1;
  has_fast_point = false;
  loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
  sequence = _sequence;
  computeWindowBRIEFPoint();
  computeBRIEFPoint(camera);
  if (!debug_image_) image.release();
}

// load previous keyframe
KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i,
                   Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
                   cv::Mat &_image, int _loop_index,
                   Eigen::Matrix<double, 8, 1> &_loop_info,
                   vector<cv::KeyPoint> &_keypoints,
                   vector<cv::KeyPoint> &_keypoints_norm,
                   vector<BRIEF::bitset> &_brief_descriptors, int image_height,
                   int image_width, std::string brief_pattern_file_path,
                   bool debug_image)
    : image_height_(image_height),
      image_width_(image_width),
      brief_pattern_file_path_(brief_pattern_file_path),
      debug_image_(debug_image) {
  time_stamp = _time_stamp;
  index = _index;
  // vio_T_w_i = _vio_T_w_i;
  // vio_R_w_i = _vio_R_w_i;
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
  if (debug_image_) {
    image = _image.clone();
    cv::resize(image, thumbnail, cv::Size(80, 60));
  }
  if (_loop_index != -1)
    has_loop = true;
  else
    has_loop = false;
  loop_index = _loop_index;
  loop_info = _loop_info;
  has_fast_point = false;
  sequence = 0;
  keypoints = _keypoints;
  keypoints_norm = _keypoints_norm;
  brief_descriptors = _brief_descriptors;
}

void KeyFrame::computeWindowBRIEFPoint() {
  BriefExtractor extractor(brief_pattern_file_path_.c_str());
  for (int i = 0; i < (int)point_2d_uv.size(); i++) {
    cv::KeyPoint key;
    key.pt = point_2d_uv[i];
    window_keypoints.push_back(key);
  }
  extractor(image, window_keypoints, window_brief_descriptors);
}

void KeyFrame::computeBRIEFPoint(camodocal::CameraPtr camera) {
  BriefExtractor extractor(brief_pattern_file_path_.c_str());
  const int fast_th = 20;  // corner detector response threshold
  if (1)
    cv::FAST(image, keypoints, fast_th, true);
  else {
    vector<cv::Point2f> tmp_pts;
    cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
    for (int i = 0; i < (int)tmp_pts.size(); i++) {
      cv::KeyPoint key;
      key.pt = tmp_pts[i];
      keypoints.push_back(key);
    }
  }
  extractor(image, keypoints, brief_descriptors);
  for (int i = 0; i < (int)keypoints.size(); i++) {
    Eigen::Vector3d tmp_p;
    camera->liftProjective(
        Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
    cv::KeyPoint tmp_norm;
    tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
    keypoints_norm.push_back(tmp_norm);
  }
}

void BriefExtractor::operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys,
                                vector<BRIEF::bitset> &descriptors) const {
  m_brief.compute(im, keys, descriptors);
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm) {
  cv::Point2f best_pt;
  int bestDist = 128;
  int bestIndex = -1;
  for (int i = 0; i < (int)descriptors_old.size(); i++) {
    int dis = HammingDis(window_descriptor, descriptors_old[i]);
    if (dis < bestDist) {
      bestDist = dis;
      bestIndex = i;
    }
  }
  // printf("best dist %d", bestDist);
  if (bestIndex != -1 && bestDist < 80) {
    best_match = keypoints_old[bestIndex].pt;
    best_match_norm = keypoints_old_norm[bestIndex].pt;
    return true;
  } else
    return false;
}

void KeyFrame::searchByBRIEFDes(
    std::vector<cv::Point2f> &matched_2d_old,
    std::vector<cv::Point2f> &matched_2d_old_norm, std::vector<uchar> &status,
    const std::vector<BRIEF::bitset> &descriptors_old,
    const std::vector<cv::KeyPoint> &keypoints_old,
    const std::vector<cv::KeyPoint> &keypoints_old_norm) {
  for (int i = 0; i < (int)window_brief_descriptors.size(); i++) {
    cv::Point2f pt(0.f, 0.f);
    cv::Point2f pt_norm(0.f, 0.f);
    if (searchInAera(window_brief_descriptors[i], descriptors_old,
                     keypoints_old, keypoints_old_norm, pt, pt_norm))
      status.push_back(1);
    else
      status.push_back(0);
    matched_2d_old.push_back(pt);
    matched_2d_old_norm.push_back(pt_norm);
  }
}

void KeyFrame::FundmantalMatrixRANSAC(
    const std::vector<cv::Point2f> &matched_2d_cur_norm,
    const std::vector<cv::Point2f> &matched_2d_old_norm,
    vector<uchar> &status) {
  int n = (int)matched_2d_cur_norm.size();
  for (int i = 0; i < n; i++) status.push_back(0);
  if (n >= 8) {
    vector<cv::Point2f> tmp_cur(n), tmp_old(n);
    for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++) {
      double FOCAL_LENGTH = 460.0;
      double tmp_x, tmp_y;
      tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + image_width_ / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + image_height_ / 2.0;
      tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

      tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + image_width_ / 2.0;
      tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + image_height_ / 2.0;
      tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
    }
    cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
  }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status, Eigen::Vector3d &PnP_T_old,
                         Eigen::Matrix3d &PnP_R_old,
                         const Eigen::Vector3d &imu_camera_translation,
                         const Eigen::Matrix3d &imu_camera_rotation) {
  // for (int i = 0; i < matched_3d.size(); i++)
  //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y,
  // matched_3d[i].z ); printf("match size %d \n", matched_3d.size());
  cv::Mat r, rvec, t, D, tmp_r;
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
  Matrix3d R_inital;
  Vector3d P_inital;
  Matrix3d R_w_c = origin_vio_R * imu_camera_rotation;
  Vector3d T_w_c = origin_vio_T + origin_vio_R * imu_camera_translation;

  R_inital = R_w_c.inverse();
  P_inital = -(R_inital * T_w_c);

  cv::eigen2cv(R_inital, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_inital, t);

  cv::Mat inliers;
  TicToc t_pnp_ransac;

  if (CV_MAJOR_VERSION < 3)
    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100,
                   10.0 / 460.0, 100, inliers);
  else {
    if (CV_MINOR_VERSION < 2)
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100,
                     sqrt(10.0 / 460.0), 0.99, inliers);
    else
      solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100,
                     10.0 / 460.0, 0.99, inliers);
  }

  for (int i = 0; i < (int)matched_2d_old_norm.size(); i++) status.push_back(0);

  for (int i = 0; i < inliers.rows; i++) {
    int n = inliers.at<int>(i);
    status[n] = 1;
  }

  cv::Rodrigues(rvec, r);
  Matrix3d R_pnp, R_w_c_old;
  cv::cv2eigen(r, R_pnp);
  R_w_c_old = R_pnp.transpose();
  Vector3d T_pnp, T_w_c_old;
  cv::cv2eigen(t, T_pnp);
  T_w_c_old = R_w_c_old * (-T_pnp);

  PnP_R_old = R_w_c_old * imu_camera_rotation.transpose();
  PnP_T_old = T_w_c_old - PnP_R_old * imu_camera_translation;
}

bool KeyFrame::findConnection(KeyFrame *old_kf,
                              vector<cv::Point2f> &matched_2d_old_norm,
                              vector<double> &matched_id,
                              const Eigen::Vector3d &imu_camera_translation,
                              const Eigen::Matrix3d &imu_camera_rotation) {
  TicToc tmp_t;
  // printf("find Connection\n");
  vector<cv::Point2f> matched_2d_cur, matched_2d_old;
  vector<cv::Point2f> matched_2d_cur_norm;
  vector<cv::Point3f> matched_3d;
  vector<uchar> status;

  matched_3d = point_3d;
  matched_2d_cur = point_2d_uv;
  matched_2d_cur_norm = point_2d_norm;
  matched_id = point_id;

  TicToc t_match;

  // printf("search by des\n");
  searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status,
                   old_kf->brief_descriptors, old_kf->keypoints,
                   old_kf->keypoints_norm);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_2d_old, status);
  reduceVector(matched_2d_cur_norm, status);
  reduceVector(matched_2d_old_norm, status);
  reduceVector(matched_3d, status);
  reduceVector(matched_id, status);
  // printf("search by des finish\n");

  status.clear();
  /*
  FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
  reduceVector(matched_2d_cur, status);
  reduceVector(matched_2d_old, status);
  reduceVector(matched_2d_cur_norm, status);
  reduceVector(matched_2d_old_norm, status);
  reduceVector(matched_3d, status);
  reduceVector(matched_id, status);
  */

  Eigen::Vector3d PnP_T_old;
  Eigen::Matrix3d PnP_R_old;
  Eigen::Vector3d relative_t;
  Quaterniond relative_q;
  double relative_yaw;
  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    status.clear();
    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old,
              imu_camera_translation, imu_camera_rotation);
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
    reduceVector(matched_3d, status);
    reduceVector(matched_id, status);

    if (debug_image_) {
      int gap = 10;
      cv::Mat gap_image(image_height_, gap, CV_8UC1, cv::Scalar(255, 255, 255));
      cv::Mat gray_img, loop_match_img;
      cv::Mat old_img = old_kf->image;
      cv::hconcat(image, gap_image, gap_image);
      cv::hconcat(gap_image, old_img, gray_img);
      cvtColor(gray_img, loop_match_img, cv::COLOR_GRAY2RGB);
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f cur_pt = matched_2d_cur[i];
        cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_old.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (image_width_ + gap);
        cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
      }
      for (int i = 0; i < (int)matched_2d_cur.size(); i++) {
        cv::Point2f old_pt = matched_2d_old[i];
        old_pt.x += (image_width_ + gap);
        cv::line(loop_match_img, matched_2d_cur[i], old_pt,
                 cv::Scalar(0, 255, 0), 2, 8, 0);
      }
      cv::Mat notation(50, image_width_ + gap + image_width_, CV_8UC3,
                       cv::Scalar(255, 255, 255));
      putText(notation,
              "current frame: " + to_string(index) +
                  "  sequence: " + to_string(sequence),
              cv::Point2f(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255),
              3);

      putText(notation,
              "previous frame: " + to_string(old_kf->index) +
                  "  sequence: " + to_string(old_kf->sequence),
              cv::Point2f(20 + image_width_ + gap, 30),
              cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
      cv::vconcat(notation, loop_match_img, loop_match_img);

      /*
      ostringstream path;
      path <<  "/home/tony-ws1/raw_data/loop_image/"
              << index << "-"
              << old_kf->index << "-" << "3pnp_match.jpg";
      cv::imwrite( path.str().c_str(), loop_match_img);
      */
      if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
        /*
        cv::imshow("loop connection",loop_match_img);
        cv::waitKey(10);
        */
        // reset thumbimage_
        {
          std::lock_guard<std::mutex> lock(m_thumbimage_);
          thumbimage_ = cv::Mat();
          cv::resize(
              loop_match_img, thumbimage_,
              cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
        }
        // sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(),
        // "bgr8", thumbimage).toImageMsg();
        // msg->header.stamp = ros::Time(time_stamp);
        // pub_match_img.publish(msg);
      }
    }
  }

  if ((int)matched_2d_cur.size() > MIN_LOOP_NUM) {
    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
    relative_q = PnP_R_old.transpose() * origin_vio_R;
    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() -
                                           Utility::R2ypr(PnP_R_old).x());
    // printf("PNP relative\n");
    // cout << "pnp relative_t " << relative_t.transpose() << endl;
    // cout << "pnp relative_yaw " << relative_yaw << endl;
    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0) {
      has_loop = true;
      loop_index = old_kf->index;
      loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
          relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
          relative_yaw;
      // if(FAST_RELOCALIZATION)
      // {
      //     sensor_msgs::PointCloud msg_match_points;
      //     msg_match_points.header.stamp = ros::Time(time_stamp);
      //     for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
      //     {
      //         geometry_msgs::Point32 p;
      //         p.x = matched_2d_old_norm[i].x;
      //         p.y = matched_2d_old_norm[i].y;
      //         p.z = matched_id[i];
      //         msg_match_points.points.push_back(p);
      //     }
      //     Eigen::Vector3d T = old_kf->T_w_i;
      //     Eigen::Matrix3d R = old_kf->R_w_i;
      //     Quaterniond Q(R);
      //     sensor_msgs::ChannelFloat32 t_q_index;
      //     t_q_index.values.push_back(T.x());
      //     t_q_index.values.push_back(T.y());
      //     t_q_index.values.push_back(T.z());
      //     t_q_index.values.push_back(Q.w());
      //     t_q_index.values.push_back(Q.x());
      //     t_q_index.values.push_back(Q.y());
      //     t_q_index.values.push_back(Q.z());
      //     t_q_index.values.push_back(index);
      //     msg_match_points.channels.push_back(t_q_index);
      // 	// TODO : move this out of class
      //     pub_match_points.publish(msg_match_points);
      // }
      return true;
    }
  }
  // printf("loop final use num %d %lf--------------- \n",
  // (int)matched_2d_cur.size(), t_match.toc());
  return false;
}

cv::Mat KeyFrame::getThumbImage() {
  std::lock_guard<std::mutex> lock(m_thumbimage_);
  return thumbimage_.clone();
}

KeyFrame::Attributes KeyFrame::getAttributes() {
  KeyFrame::Attributes attr;
  attr.index = index;
  attr.time_stamp = time_stamp;
  attr.sequence = sequence;
  attr.has_loop = has_loop;
  attr.loop_index = loop_index;
  attr.position = T_w_i;
  attr.rotation = R_w_i;
  attr.vio_position = vio_T_w_i;
  attr.vio_rotation = vio_R_w_i;
  return attr;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b) {
  BRIEF::bitset xor_of_bitset = a ^ b;
  int dis = xor_of_bitset.count();
  return dis;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) {
  _T_w_i = vio_T_w_i;
  _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) {
  _T_w_i = T_w_i;
  _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i,
                          const Eigen::Matrix3d &_R_w_i) {
  T_w_i = _T_w_i;
  R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i,
                             const Eigen::Matrix3d &_R_w_i) {
  vio_T_w_i = _T_w_i;
  vio_R_w_i = _R_w_i;
  T_w_i = vio_T_w_i;
  R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT() {
  return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ() {
  return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5),
                            loop_info(6));
}

double KeyFrame::getLoopRelativeYaw() { return loop_info(7); }

void KeyFrame::updateLoop(const Eigen::Matrix<double, 8, 1> &_loop_info) {
  if (abs(_loop_info(7)) < 30.0 &&
      Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
    // printf("update loop info\n");
    loop_info = _loop_info;
  }
}

BriefExtractor::BriefExtractor(const std::string &pattern_file) {
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}
