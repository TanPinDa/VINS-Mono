/**
 * @file pose_graph.cpp
 * @brief
 * @date 01-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include "pose_graph/details/pose_graph.hpp"

#include <fstream>
#include <vector>

#include <ceres/ceres.h>

#include "pose_graph/utility/tic_toc.h"

namespace pose_graph {
namespace {
template <typename T>
T NormalizeAngle(const T& angle_degrees) {
  if (angle_degrees > T(180.0))
    return angle_degrees - T(360.0);
  else if (angle_degrees < T(-180.0))
    return angle_degrees + T(360.0);
  else
    return angle_degrees;
};

struct AngleManifoldFunctor {
  template <typename T>
  bool Plus(const T* theta_radians, const T* delta_theta_radians,
            T* theta_radians_plus_delta) const {
    *theta_radians_plus_delta =
        NormalizeAngle(*theta_radians + *delta_theta_radians);

    return true;
  }

  template <typename T>
  bool Minus(const T* theta_y_radians, const T* theta_x_radians,
             T* theta_y_minus_x_radians) const {
    *theta_y_minus_x_radians =
        NormalizeAngle(*theta_y_radians - *theta_x_radians);

    return true;
  }
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll,
                                  T R[9]) {
  T y = yaw / T(180.0) * T(M_PI);
  T p = pitch / T(180.0) * T(M_PI);
  T r = roll / T(180.0) * T(M_PI);

  R[0] = cos(y) * cos(p);
  R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
  R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
  R[3] = sin(y) * cos(p);
  R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
  R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
  R[6] = -sin(p);
  R[7] = cos(p) * sin(r);
  R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9]) {
  inv_R[0] = R[0];
  inv_R[1] = R[3];
  inv_R[2] = R[6];
  inv_R[3] = R[1];
  inv_R[4] = R[4];
  inv_R[5] = R[7];
  inv_R[6] = R[2];
  inv_R[7] = R[5];
  inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3]) {
  r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
  r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
  r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};

struct FourDOFError {
  FourDOFError(double t_x, double t_y, double t_z, double relative_yaw,
               double pitch_i, double roll_i)
      : t_x(t_x),
        t_y(t_y),
        t_z(t_z),
        relative_yaw(relative_yaw),
        pitch_i(pitch_i),
        roll_i(roll_i) {}

  template <typename T>
  bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j,
                  const T* tj, T* residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    // euler to rotation
    T w_R_i[9];
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
    // rotation transpose
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    // rotation matrix rotate point
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x));
    residuals[1] = (t_i_ij[1] - T(t_y));
    residuals[2] = (t_i_ij[2] - T(t_z));
    residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw));

    return true;
  }

  static ceres::CostFunction* Create(const double t_x, const double t_y,
                                     const double t_z,
                                     const double relative_yaw,
                                     const double pitch_i,
                                     const double roll_i) {
    return (new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
        new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
  }

  double t_x, t_y, t_z;
  double relative_yaw, pitch_i, roll_i;
};

struct FourDOFWeightError {
  FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw,
                     double pitch_i, double roll_i)
      : t_x(t_x),
        t_y(t_y),
        t_z(t_z),
        relative_yaw(relative_yaw),
        pitch_i(pitch_i),
        roll_i(roll_i) {
    weight = 1;
  }

  template <typename T>
  bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j,
                  const T* tj, T* residuals) const {
    T t_w_ij[3];
    t_w_ij[0] = tj[0] - ti[0];
    t_w_ij[1] = tj[1] - ti[1];
    t_w_ij[2] = tj[2] - ti[2];

    // euler to rotation
    T w_R_i[9];
    YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
    // rotation transpose
    T i_R_w[9];
    RotationMatrixTranspose(w_R_i, i_R_w);
    // rotation matrix rotate point
    T t_i_ij[3];
    RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

    residuals[0] = (t_i_ij[0] - T(t_x)) * T(weight);
    residuals[1] = (t_i_ij[1] - T(t_y)) * T(weight);
    residuals[2] = (t_i_ij[2] - T(t_z)) * T(weight);
    residuals[3] = NormalizeAngle((yaw_j[0] - yaw_i[0] - T(relative_yaw))) *
                   T(weight) / T(10.0);

    return true;
  }

  static ceres::CostFunction* Create(const double t_x, const double t_y,
                                     const double t_z,
                                     const double relative_yaw,
                                     const double pitch_i,
                                     const double roll_i) {
    return (new ceres::AutoDiffCostFunction<FourDOFWeightError, 4, 1, 3, 1, 3>(
        new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
  }

  double t_x, t_y, t_z;
  double relative_yaw, pitch_i, roll_i;
  double weight;
};
}  // namespace

PoseGraph::PoseGraph(const PoseGraphConfig& config) : config_(config) {
  LoadVocabulary();
}

std::vector<std::shared_ptr<KeyFrame>> PoseGraph::Load() {
  // Load previously saved pose graph from file
  TicToc clock;
  FILE* pFile;
  std::string file_path = config_.saved_pose_graph_dir + "pose_graph.txt";
  printf("loading pose graph from: %s \n", file_path.c_str());
  printf("pose graph loading...\n");
  pFile = fopen(file_path.c_str(), "r");
  if (pFile == NULL) {
    printf(
        "load previous pose graph error: wrong previous pose graph path or no "
        "previous pose graph \n the system will start with new pose graph \n");
    return std::vector<std::shared_ptr<KeyFrame>>();
  }
  int index;
  double time_stamp;
  double VIO_Tx, VIO_Ty, VIO_Tz;
  double PG_Tx, PG_Ty, PG_Tz;
  double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
  double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
  double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
  double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
  int loop_index;
  int keypoints_num;
  Eigen::Matrix<double, 8, 1> loop_info;
  std::vector<std::shared_ptr<KeyFrame>> loaded_keyframes;

  // TODO for Kee Jin: Review using fscanf to load data from file
  while (fscanf(pFile,
                "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
                "%lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d",
                &index, &time_stamp, &VIO_Tx, &VIO_Ty, &VIO_Tz, &PG_Tx, &PG_Ty,
                &PG_Tz, &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, &PG_Qw, &PG_Qx,
                &PG_Qy, &PG_Qz, &loop_index, &loop_info_0, &loop_info_1,
                &loop_info_2, &loop_info_3, &loop_info_4, &loop_info_5,
                &loop_info_6, &loop_info_7, &keypoints_num) != EOF) {
    cv::Mat image;
    std::string image_path, descriptor_path;
    if (config_.save_debug_image) {
      image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
      image = cv::imread(image_path.c_str(), 0);
    }

    Eigen::Vector3d vio_translation(VIO_Tx, VIO_Ty, VIO_Tz);
    Eigen::Quaterniond vio_quarternion;
    Eigen::Vector3d pose_graph_translation(PG_Tx, PG_Ty, PG_Tz);
    vio_quarternion.w() = VIO_Qw;
    vio_quarternion.x() = VIO_Qx;
    vio_quarternion.y() = VIO_Qy;
    vio_quarternion.z() = VIO_Qz;
    Eigen::Quaterniond pose_graph_quarternion;
    pose_graph_quarternion.w() = PG_Qw;
    pose_graph_quarternion.x() = PG_Qx;
    pose_graph_quarternion.y() = PG_Qy;
    pose_graph_quarternion.z() = PG_Qz;
    Eigen::Matrix3d vio_rotation, pose_graph_rotation;
    vio_rotation = vio_quarternion.toRotationMatrix();
    pose_graph_rotation = pose_graph_quarternion.toRotationMatrix();
    Eigen::Matrix<double, 8, 1> loop_info;
    loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3,
        loop_info_4, loop_info_5, loop_info_6, loop_info_7;

    if (loop_index != -1)
      if (earliest_loop_index > loop_index || earliest_loop_index == -1) {
        earliest_loop_index = loop_index;
      }

    // load keypoints, brief_descriptors
    std::string brief_path =
        config_.saved_pose_graph_dir + std::to_string(index) + "_briefdes.dat";
    std::ifstream brief_file(brief_path, std::ios::binary);
    std::string keypoints_path =
        config_.saved_pose_graph_dir + to_string(index) + "_keypoints.txt";
    FILE* keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "r");
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypoints_norm;
    std::vector<BRIEF::bitset> brief_descriptors;
    for (int i = 0; i < keypoints_num; i++) {
      BRIEF::bitset tmp_des;
      brief_file >> tmp_des;
      brief_descriptors.push_back(tmp_des);
      cv::KeyPoint tmp_keypoint;
      cv::KeyPoint tmp_keypoint_norm;
      double p_x, p_y, p_x_norm, p_y_norm;
      if (!fscanf(keypoints_file, "%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm,
                  &p_y_norm)) {
        printf(" fail to load pose graph \n");
      }
      tmp_keypoint.pt.x = p_x;
      tmp_keypoint.pt.y = p_y;
      tmp_keypoint_norm.pt.x = p_x_norm;
      tmp_keypoint_norm.pt.y = p_y_norm;
      keypoints.push_back(tmp_keypoint);
      keypoints_norm.push_back(tmp_keypoint_norm);
    }
    brief_file.close();
    fclose(keypoints_file);

    // Create and load keyframe
    std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(
        time_stamp, index, vio_translation, vio_rotation,
        pose_graph_translation, pose_graph_rotation, image, loop_index,
        loop_info, keypoints, keypoints_norm, brief_descriptors);
    LoadKeyFrame(keyframe);
    loaded_keyframes.push_back(keyframe);
  }
  fclose(pFile);
  printf("pose graph loaded, time cost: %f s\n", clock.toc() / 1000);
  base_sequence = 0;
  return loaded_keyframes;
}

void PoseGraph::Save() {
  // Save keyframes to file
  std::lock_guard<std::mutex> lock(keyframes_mutex_);
  TicToc clock;
  FILE* pFile;
  printf("saving pose graph to: %s \n",
         (config_.saved_pose_graph_dir + "pose_graph.txt").c_str());
  std::string file_path = config_.saved_pose_graph_dir + "pose_graph.txt";
  pFile = fopen(file_path.c_str(), "w");
  for (auto keyframe : keyframes_) {
    std::string image_path, descriptor_path, brief_path, keypoints_path;
    if (config_.save_debug_image) {
      image_path = config_.saved_pose_graph_dir +
                   std::to_string(keyframe->index) + "_image.png";
      cv::imwrite(image_path.c_str(), keyframe->image);
    }
    Quaterniond vio_tmp_quarternion{keyframe->vio_R_w_i};
    Quaterniond pose_graph_tmp_quarternion{keyframe->R_w_i};
    Vector3d vio_tmp_translation = keyframe->vio_T_w_i;
    Vector3d pose_graph_tmp_translation = keyframe->T_w_i;

    fprintf(pFile,
            " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f "
            "%f %f %f %f %d\n",
            keyframe->index, keyframe->time_stamp, vio_tmp_translation.x(),
            vio_tmp_translation.y(), vio_tmp_translation.z(),
            pose_graph_tmp_translation.x(), pose_graph_tmp_translation.y(),
            pose_graph_tmp_translation.z(), vio_tmp_quarternion.w(),
            vio_tmp_quarternion.x(), vio_tmp_quarternion.y(),
            vio_tmp_quarternion.z(), pose_graph_tmp_quarternion.w(),
            pose_graph_tmp_quarternion.x(), pose_graph_tmp_quarternion.y(),
            pose_graph_tmp_quarternion.z(), keyframe->loop_index,
            keyframe->loop_info(0), keyframe->loop_info(1),
            keyframe->loop_info(2), keyframe->loop_info(3),
            keyframe->loop_info(4), keyframe->loop_info(5),
            keyframe->loop_info(6), keyframe->loop_info(7),
            (int)keyframe->keypoints.size());

    // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints
    // vector<BRIEF::bitset> brief_descriptors;
    assert(keyframe->keypoints.size() == keyframe->brief_descriptors.size());
    brief_path = config_.saved_pose_graph_dir +
                 std::to_string(keyframe->index) + "_briefdes.dat";
    std::ofstream brief_file(brief_path, std::ios::binary);
    keypoints_path = config_.saved_pose_graph_dir +
                     std::to_string(keyframe->index) + "_keypoints.txt";
    FILE* keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "w");
    for (int i = 0; i < (int)keyframe->keypoints.size(); i++) {
      brief_file << keyframe->brief_descriptors[i] << endl;
      fprintf(keypoints_file, "%f %f %f %f\n", keyframe->keypoints[i].pt.x,
              keyframe->keypoints[i].pt.y, keyframe->keypoints_norm[i].pt.x,
              keyframe->keypoints_norm[i].pt.y);
    }
    brief_file.close();
    fclose(keypoints_file);
  }
  fclose(pFile);

  printf("pose graph saved, time cost: %f s\n", clock.toc() / 1000);
}

void PoseGraph::AddKeyFrame(std::shared_ptr<KeyFrame> current_kf) {
  // Add keyframe to the pose graph
}

void PoseGraph::LoadKeyFrame(std::shared_ptr<KeyFrame> current_kf) {
  // Load keyframe from the pose graph
}

int PoseGraph::GetCurrentSequenceCount() const {
  return current_sequence_count_;
}

PoseGraph::Drift PoseGraph::GetDrift() const { return drift_; }

int PoseGraph::DetectLoopClosure(std::shared_ptr<KeyFrame> current_kf) {
  // Detect loop closure
  return -1;
}

void PoseGraph::LoadVocabulary() {
  // Load vocabulary from file
  vocabulary_ = std::make_unique<BriefVocabulary>(config_.vocabulary_path);
  db_.setVocabulary(*vocabulary_);
}

void PoseGraph::AddKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe,
                                   int frame_index) {
  // Add keyframe into vocabulary
}

void PoseGraph::Optimize4DoF() {
  // Optimize 4 degrees of freedom
}
}  // namespace pose_graph
