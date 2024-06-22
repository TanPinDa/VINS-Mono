/**
 * @file pose_graph.cpp
 * @brief
 * @date 01-06-2024
 *
 * @copyright Copyright (c) 2024 Cheo Kee Jin.
 */

#include "pose_graph/pose_graph.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>

#include <ceres/ceres.h>

#include "pose_graph/utility/tic_toc.h"

namespace pose_graph
{
PoseGraph::PoseGraph()
{
}

void PoseGraph::Initialize(const PoseGraphConfig &config,
                            camodocal::CameraPtr camera)
{
  config_ = config;
  camera_ = camera;
  LoadVocabulary();
  StartOptimizationThread();
}

PoseGraph::~PoseGraph()
{
  keep_running_ = false;
  if (optimization_thread_.joinable())
  {
    optimization_thread_.join();
  }
}

void PoseGraph::RegisterEventObserver(
    std::shared_ptr<PoseGraphEventObserver> event_observer)
{
  event_observer_ = event_observer;
}

void PoseGraph::LoadVocabulary()
{
  // Load vocabulary from file
  vocabulary_ = std::make_unique<BriefVocabulary>(config_.vocabulary_path);
  db_.setVocabulary(*vocabulary_);
}

  void PoseGraph::AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe,
                              int &old_keyframe_loop_index,
                              std::vector<cv::Point2f> &matched_2d_old_norm,
                              std::vector<double> &matched_id)
  {
    // shift to base frame
    Vector3d vio_current_position;
    Matrix3d vio_current_rotation;
    if (current_sequence_count_ != current_keyframe->sequence)
    {
      current_sequence_count_++;
      sequence_loop_flags_.push_back(false);
      world_vio_.translation = Eigen::Vector3d(0, 0, 0);
      world_vio_.rotation = Eigen::Matrix3d::Identity();
      {
        std::lock_guard<std::mutex> lock(drift_mutex_);
        drift_.translation = Eigen::Vector3d(0, 0, 0);
        drift_.rotation = Eigen::Matrix3d::Identity();
      }
    }

    current_keyframe->getVioPose(vio_current_position, vio_current_rotation);
    vio_current_position =
        world_vio_.rotation * vio_current_position + world_vio_.translation;
    vio_current_rotation = world_vio_.rotation * vio_current_rotation;
    current_keyframe->updateVioPose(vio_current_position, vio_current_rotation);
    // Assign the current global_keyframe_index_counter_, then increment it
    current_keyframe->index = global_keyframe_index_counter_++;
    old_keyframe_loop_index = -1;
    if (config_.detect_loop_closure)
    {
      old_keyframe_loop_index = DetectLoopClosure(current_keyframe);
    }
    else
    {
      AddKeyFrameIntoVoc(current_keyframe);
    }

    KeyFrame *old_keyframe = nullptr;
    if (old_keyframe_loop_index != -1)
    {
      old_keyframe = GetKeyFrame(old_keyframe_loop_index).get();
      if (old_keyframe == nullptr)
      {
        throw std::runtime_error("AddKeyFrame(): loop index not found");
      }

      if (current_keyframe->findConnection(
              old_keyframe, matched_2d_old_norm, matched_id,
              imu_camera_pose_.translation, imu_camera_pose_.rotation))
      {
        if (earliest_loop_index > old_keyframe_loop_index || earliest_loop_index == -1)
        {
          earliest_loop_index = old_keyframe_loop_index;
        }

        Vector3d old_world_position, current_world_position, current_vio_position;
        Matrix3d old_world_rotation, current_world_rotation, current_vio_rotation;
        old_keyframe->getVioPose(old_world_position, old_world_rotation);
        current_keyframe->getVioPose(current_vio_position, current_vio_rotation);

        Vector3d relative_translation;
        Quaterniond relative_quarternion;
        relative_translation = current_keyframe->getLoopRelativeT();
        relative_quarternion =
            (current_keyframe->getLoopRelativeQ()).toRotationMatrix();
        current_world_position =
            old_world_rotation * relative_translation + old_world_position;
        current_world_rotation = old_world_rotation * relative_quarternion;
        double shift_yaw;
        Matrix3d shift_rotation;
        Vector3d shift_translation;
        shift_yaw = Utility::R2ypr(current_world_rotation).x() -
                    Utility::R2ypr(current_vio_rotation).x();
        shift_rotation = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
        shift_translation =
            current_world_position - current_world_rotation *
                                         current_vio_rotation.transpose() *
                                         current_vio_position;
        // shift vio pose of whole sequence to the world frame
        if (old_keyframe->sequence != current_keyframe->sequence &&
            sequence_loop_flags_[current_keyframe->sequence] == 0)
        {
          world_vio_.rotation = shift_rotation;
          world_vio_.translation = shift_translation;
          current_vio_position =
              world_vio_.rotation * current_vio_position + world_vio_.translation;
          current_vio_rotation = world_vio_.rotation * current_vio_rotation;
          current_keyframe->updateVioPose(current_vio_position,
                                          current_vio_rotation);
          for (auto keyframe : keyframes_)
          {
            if (keyframe->sequence == current_keyframe->sequence)
            {
              Vector3d current_vio_position;
              Matrix3d current_vio_rotation;
              keyframe->getVioPose(current_vio_position, current_vio_rotation);
              current_vio_position = world_vio_.rotation * current_vio_position +
                                     world_vio_.translation;
              current_vio_rotation = world_vio_.rotation * current_vio_rotation;
              keyframe->updateVioPose(current_vio_position, current_vio_rotation);
            }
          }
          sequence_loop_flags_[current_keyframe->sequence] = 1;
        }
        {
          std::lock_guard<std::mutex> lock(optimize_buf_mutex_);
          optimize_buf_.push(current_keyframe->index);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(keyframes_mutex_);
      Eigen::Vector3d position;
      Eigen::Matrix3d rotation;
      current_keyframe->getVioPose(position, rotation);
      position = drift_.rotation * position + drift_.translation;
      rotation = drift_.rotation * rotation;
      current_keyframe->updatePose(position, rotation);
      keyframes_.push_back(current_keyframe);
    }
  }
 void PoseGraph::AddKeyFrame(std::shared_ptr<KeyFrame> current_keyframe)
  {
    int old_keyframe_loop_index = -1;
    std::vector<cv::Point2f> matched_2d_old_norm;
    std::vector<double> matched_id;
    AddKeyFrame(current_keyframe, old_keyframe_loop_index, matched_2d_old_norm, matched_id);

    if (!event_observer_)
    {
      // If no observer is registered, the rest of the function is not needed.
      return;
    }

    if (old_keyframe_loop_index != -1)
    {
      cv::Mat image = current_keyframe->getThumbImage();
      auto old_keyframe = GetKeyFrame(old_keyframe_loop_index);
      event_observer_->OnKeyFrameConnectionFound(
          current_keyframe->getAttributes(), old_keyframe->getAttributes(),
          matched_2d_old_norm, matched_id, image);
    }

    // Show sequential edge
    auto attributes = GetKeyFrameAttributes();
    std::vector<KeyFrame::Attributes>::reverse_iterator rit = attributes.rbegin();
    Vector3d P;
    Matrix3d R;
    current_keyframe->getPose(P, R);
    for (int i = 0; i < 4; i++)
    {
      if (rit == attributes.rend())
        break;
      Vector3d connected_P;
      Matrix3d connected_R;
      if ((*rit).sequence == current_keyframe->sequence)
      {
        event_observer_->OnNewSequentialEdge(P, (*rit).position);
      }
      rit++;
    }

    // Show loop edge
    if (current_keyframe->has_loop)
    {
      // printf("has loop \n");
      KeyFrame::Attributes connected_kf_attributes =
          GetKeyFrameAttribute(current_keyframe->loop_index);
      Vector3d P0;
      Matrix3d R0;
      // current_keyframe->getVioPose(P0, R0);
      current_keyframe->getPose(P0, R0);
      if (current_keyframe->sequence > 0)
      {
        // printf("add loop into visual \n");
        event_observer_->OnNewLoopEdge(P0, connected_kf_attributes.position);
      }
    }

    // Generic callback
    event_observer_->OnKeyFrameAdded(current_keyframe->getAttributes());
}

  void PoseGraph::LoadKeyFrame(std::shared_ptr<KeyFrame> current_keyframe,
                               KeyFrame *old_keyframe,
                               std::vector<cv::Point2f> &matched_2d_old_norm,
                               std::vector<double> &matched_id)
  {
    // Assign the current global_keyframe_index_counter_, then increment it
    current_keyframe->index = global_keyframe_index_counter_++;
    int loop_index = -1;
    if (config_.detect_loop_closure)
    {
      loop_index = DetectLoopClosure(current_keyframe);
    }
    else
    {
      AddKeyFrameIntoVoc(current_keyframe);
    }

    if (loop_index != -1)
    {
      printf(" %d detected loop closure with %d \n", current_keyframe->index,
             loop_index);
      old_keyframe = GetKeyFrame(loop_index).get();
      if (old_keyframe == nullptr)
      {
        throw std::runtime_error("LoadKeyFrame(): loop index not found");
      }
      std::vector<cv::Point2f> matched_2d_old_norm;
      std::vector<double> matched_id;
      if (current_keyframe->findConnection(
              old_keyframe, matched_2d_old_norm, matched_id,
              imu_camera_pose_.translation, imu_camera_pose_.rotation))
      {
        if (earliest_loop_index > loop_index || earliest_loop_index == -1)
        {
          earliest_loop_index = loop_index;
        }
        {
          std::lock_guard<std::mutex> lock(optimize_buf_mutex_);
          optimize_buf_.push(current_keyframe->index);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(keyframes_mutex_);
      keyframes_.push_back(current_keyframe);
    }
}

std::shared_ptr<KeyFrame> PoseGraph::GetKeyFrame(int index)
{
  // TODO for Kee Jin: Check if a mutex is needed here to access keyframes_

  // Get keyframe from the pose graph
  auto it = std::find_if(keyframes_.begin(), keyframes_.end(),
                          [index](const std::shared_ptr<KeyFrame> &kf)
                          {
                            return kf->index == index;
                          });

    if (it != keyframes_.end())
      return *it;
    else
      return nullptr;
}

  int PoseGraph::DetectLoopClosure(std::shared_ptr<KeyFrame> current_keyframe)
  {
    // Add image to pool for visualization
    cv::Mat compressed_image;
    if (config_.save_debug_image)
    {
      int feature_num = current_keyframe->keypoints.size();
      cv::resize(current_keyframe->image, compressed_image, cv::Size(376, 240));
      cv::putText(compressed_image, "Feature Num: " + std::to_string(feature_num),
                  cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(255));
      image_pool_[current_keyframe->index] = compressed_image;
    }

    DBoW2::QueryResults ret;
    db_.query(current_keyframe->brief_descriptors, ret, 4,
              current_keyframe->index - 50);
    db_.add(current_keyframe->brief_descriptors);

    bool loop_detected = false;
    if (config_.save_debug_image)
    {
      // NOTE: loop_result seems to be unused
      cv::Mat loop_result;
      loop_result = compressed_image.clone();
      if (ret.size() > 0)
      {
        putText(loop_result, "neighbour score:" + to_string(ret[0].Score),
                cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255));
      }

      for (unsigned int i = 0; i < ret.size(); i++)
      {
        int tmp_index = ret[i].Id;
        auto it = image_pool_.find(tmp_index);
        cv::Mat tmp_image = (it->second).clone();
        cv::putText(tmp_image,
                    "index:  " + to_string(tmp_index) +
                        "loop score:" + to_string(ret[i].Score),
                    cv::Point2f(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255));
        cv::hconcat(loop_result, tmp_image, loop_result);
      }
    }

    // a good match with its nerghbour
    if (ret.size() >= 1 && ret[0].Score > 0.05)
    {
      for (unsigned int i = 1; i < ret.size(); i++)
      {
        // if (ret[i].Score > ret[0].Score * 0.3)
        if (ret[i].Score > 0.015)
        {
          loop_detected = true;
          int tmp_index = ret[i].Id;
        }
      }
    }

    if (loop_detected && current_keyframe->index > 50)
    {
      int min_index = -1;
      for (unsigned int i = 0; i < ret.size(); i++)
      {
        if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
          min_index = ret[i].Id;
      }
      return min_index;
    }
    else
    {
      return -1;
    }
  }

  void PoseGraph::AddKeyFrameIntoVoc(std::shared_ptr<KeyFrame> keyframe)
  {
    // add image into image_pool for visualization
    if (config_.save_debug_image)
    {
      cv::Mat compressed_image;
      int feature_num = keyframe->keypoints.size();
      cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
      putText(compressed_image, "feature_num:" + to_string(feature_num),
              cv::Point2f(10, 10), cv::FONT_HERSHEY_SIMPLEX, 0.4,
              cv::Scalar(255));
      image_pool_[keyframe->index] = compressed_image;
    }

    db_.add(keyframe->brief_descriptors);
  }

  void PoseGraph::Optimize4DoF()
  {
    int current_index = -1;
    int first_looped_index = -1;
    {
      std::lock_guard<std::mutex> lock(optimize_buf_mutex_);
      while (!optimize_buf_.empty())
      {
        current_index = optimize_buf_.front();
        first_looped_index = earliest_loop_index;
        optimize_buf_.pop();
      }
    }
    if (current_index != -1)
    {
      // printf("optimize pose graph \n");
      ceres::Problem problem;
      int i = 0;
      list<std::shared_ptr<KeyFrame>>::iterator it;
      int max_length = current_index + 1;

      // w^t_i   w^q_i
      double translation_array[max_length][3];
      Quaterniond quarternion_array[max_length];
      double euler_array[max_length][3];
      double sequence_array[max_length];
      std::shared_ptr<KeyFrame> current_keyframe;

      {
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        current_keyframe = GetKeyFrame(current_index);
        if (!current_keyframe)
        {
          throw std::runtime_error("Optimize4DoF(): current keyframe not found");
        }

        ceres::LossFunction *loss_function;
        loss_function = new ceres::HuberLoss(0.1);
        // loss_function = new ceres::CauchyLoss(1.0);
        ceres::Manifold *angle_manifold =
            new ceres::AutoDiffManifold<AngleManifoldFunctor, 1, 1>;

        for (it = keyframes_.begin(); it != keyframes_.end(); it++)
        {
          if ((*it)->index < first_looped_index)
            continue;
          (*it)->local_index = i;
          Quaterniond tmp_q;
          Matrix3d tmp_r;
          Vector3d tmp_t;
          (*it)->getVioPose(tmp_t, tmp_r);
          tmp_q = tmp_r;
          translation_array[i][0] = tmp_t(0);
          translation_array[i][1] = tmp_t(1);
          translation_array[i][2] = tmp_t(2);
          quarternion_array[i] = tmp_q;

          Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
          euler_array[i][0] = euler_angle.x();
          euler_array[i][1] = euler_angle.y();
          euler_array[i][2] = euler_angle.z();

          sequence_array[i] = (*it)->sequence;

          problem.AddParameterBlock(euler_array[i], 1, angle_manifold);
          problem.AddParameterBlock(translation_array[i], 3);

          if ((*it)->index == first_looped_index || (*it)->sequence == 0)
          {
            problem.SetParameterBlockConstant(euler_array[i]);
            problem.SetParameterBlockConstant(translation_array[i]);
          }

          // add edge
          for (int j = 1; j < 5; j++)
          {
            if (i - j >= 0 && sequence_array[i] == sequence_array[i - j])
            {
              Vector3d euler_conncected =
                  Utility::R2ypr(quarternion_array[i - j].toRotationMatrix());
              Vector3d relative_t(
                  translation_array[i][0] - translation_array[i - j][0],
                  translation_array[i][1] - translation_array[i - j][1],
                  translation_array[i][2] - translation_array[i - j][2]);
              relative_t = quarternion_array[i - j].inverse() * relative_t;
              double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
              ceres::CostFunction *cost_function = FourDOFError::Create(
                  relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw,
                  euler_conncected.y(), euler_conncected.z());
              problem.AddResidualBlock(cost_function, NULL, euler_array[i - j],
                                       translation_array[i - j], euler_array[i],
                                       translation_array[i]);
            }
          }

          // add loop edge

          if ((*it)->has_loop)
          {
            assert((*it)->loop_index >= first_looped_index);
            int connected_index = GetKeyFrame((*it)->loop_index)->local_index;
            Vector3d euler_conncected = Utility::R2ypr(
                quarternion_array[connected_index].toRotationMatrix());
            Vector3d relative_t;
            relative_t = (*it)->getLoopRelativeT();
            double relative_yaw = (*it)->getLoopRelativeYaw();
            ceres::CostFunction *cost_function = FourDOFWeightError::Create(
                relative_t.x(), relative_t.y(), relative_t.z(), relative_yaw,
                euler_conncected.y(), euler_conncected.z());
            problem.AddResidualBlock(cost_function, loss_function,
                                     euler_array[connected_index],
                                     translation_array[connected_index],
                                     euler_array[i], translation_array[i]);
          }

          if ((*it)->index == current_index)
            break;
          i++;
        }
      }

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      // options.minimizer_progress_to_stdout = true;
      // options.max_solver_time_in_seconds = SOLVER_TIME * 3;
      options.max_num_iterations = 5;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      // std::cout << summary.BriefReport() << "\n";

      // printf("pose optimization time: %f \n", tmp_t.toc());
      /*
      for (int j = 0 ; j < i; j++)
      {
          printf("optimize i: %d p: %f, %f, %f\n", j, translation_array[j][0],
      translation_array[j][1], translation_array[j][2] );
      }
      */
      {
        std::lock_guard<std::mutex> lock(keyframes_mutex_);
        i = 0;
        for (it = keyframes_.begin(); it != keyframes_.end(); it++)
        {
          if ((*it)->index < first_looped_index)
            continue;
          Quaterniond tmp_q;
          tmp_q = Utility::ypr2R(
              Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
          Vector3d tmp_t =
              Vector3d(translation_array[i][0], translation_array[i][1],
                       translation_array[i][2]);
          Matrix3d tmp_r = tmp_q.toRotationMatrix();
          (*it)->updatePose(tmp_t, tmp_r);

          if ((*it)->index == current_index)
            break;
          i++;
        }

        Vector3d cur_t, vio_t;
        Matrix3d cur_r, vio_r;
        current_keyframe->getPose(cur_t, cur_r);
        current_keyframe->getVioPose(vio_t, vio_r);
        {
          std::lock_guard<std::mutex> lock(drift_mutex_);
          drift_.yaw = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
          drift_.rotation = Utility::ypr2R(Vector3d(drift_.yaw, 0, 0));
          drift_.translation = cur_t - drift_.rotation * vio_t;
        }
        // cout << "t_drift " << drift_.translation.transpose() << endl;
        // cout << "r_drift " << Utility::R2ypr(drift_.rotation).transpose() <<
        // endl; cout << "yaw drift " << drift_.yaw << endl;

        it++;
        for (; it != keyframes_.end(); it++)
        {
          Vector3d position;
          Matrix3d rotation;
          (*it)->getVioPose(position, rotation);
          position = drift_.rotation * position + drift_.translation;
          rotation = drift_.rotation * rotation;
          (*it)->updatePose(position, rotation);
        }
        // updatePath();
      }
    }
}

void PoseGraph::Save()
{
  // Save keyframes to file
  std::lock_guard<std::mutex> lock(keyframes_mutex_);
  TicToc clock;
  FILE *pFile;
  printf("saving pose graph to: %s \n",
          (config_.saved_pose_graph_dir + "pose_graph.txt").c_str());
  std::string file_path = config_.saved_pose_graph_dir + "pose_graph.txt";
    pFile = fopen(file_path.c_str(), "w");
    for (auto keyframe : keyframes_)
    {
      std::string image_path, descriptor_path, brief_path, keypoints_path;
      if (config_.save_debug_image)
      {
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
      FILE *keypoints_file;
      keypoints_file = fopen(keypoints_path.c_str(), "w");
      for (int i = 0; i < (int)keyframe->keypoints.size(); i++)
      {
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

    if (event_observer_)
    {
      event_observer_->OnPoseGraphSaved();
    }
}
bool PoseGraph::Load()
{
    // Load previously saved pose graph from file
    TicToc clock;
    FILE *pFile;
    std::string file_path = config_.saved_pose_graph_dir + "pose_graph.txt";
    printf("loading pose graph from: %s \n", file_path.c_str());
    printf("pose graph loading...\n");
    pFile = fopen(file_path.c_str(), "r");
    if (pFile == NULL)
    {
      printf(
          "load previous pose graph error: wrong previous pose graph path or no "
          "previous pose graph \n the system will start with new pose graph \n");
      return false;
    }

    KeyFrame::Attributes old_kf_attribute;
    KeyFrame::Attributes current_kf_attribute;
    std::vector<cv::Point2f> matched_2d_old_norm;
    std::vector<double> matched_id;
    cv::Mat current_kf_thumb_image;
    int count = 0;
    while (LoadSingleConfigEntry(pFile, old_kf_attribute, current_kf_attribute,
                                 matched_2d_old_norm, matched_id,
                                 current_kf_thumb_image))
    {
      if (old_kf_attribute.time_stamp >= 0.0)
      {
        if (event_observer_)
        {
          event_observer_->OnKeyFrameConnectionFound(
              current_kf_attribute, old_kf_attribute, matched_2d_old_norm,
              matched_id, current_kf_thumb_image);
        }
      }
      if (event_observer_)
      {
        event_observer_->OnKeyFrameLoaded(current_kf_attribute, count);
      }
      count++;
    }

    fclose(pFile);
    printf("pose graph loaded, time cost: %f s\n", clock.toc() / 1000);

    // Generic callback
    if (event_observer_)
    {
      event_observer_->OnPoseGraphLoaded();
    }

    return true;
  }

  bool PoseGraph::LoadSingleConfigEntry(
      FILE *pFile, KeyFrame::Attributes &old_kf_attribute,
      KeyFrame::Attributes &current_kf_attribute,
      std::vector<cv::Point2f> &matched_2d_old_norm,
      std::vector<double> &matched_id, cv::Mat &current_kf_thumb_image)
  {
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
    std::vector<std::shared_ptr<KeyFrame>> loaded_keyframes;

    // TODO for Kee Jin: Review using fscanf to load data from file
    if (fscanf(pFile,
               "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
               "%lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d",
               &index, &time_stamp, &VIO_Tx, &VIO_Ty, &VIO_Tz, &PG_Tx, &PG_Ty,
               &PG_Tz, &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, &PG_Qw, &PG_Qx, &PG_Qy,
               &PG_Qz, &loop_index, &loop_info_0, &loop_info_1, &loop_info_2,
               &loop_info_3, &loop_info_4, &loop_info_5, &loop_info_6,
               &loop_info_7, &keypoints_num) == EOF)
    {
      // reached end of file
      base_sequence = 0;
      return false;
    }

    cv::Mat image;
    std::string image_path, descriptor_path;
    if (config_.save_debug_image)
    {
      image_path = config_.saved_pose_graph_dir + to_string(index) + "_image.png";
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
    loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4,
        loop_info_5, loop_info_6, loop_info_7;

    if (loop_index != -1)
    {
      if (earliest_loop_index > loop_index || earliest_loop_index == -1)
      {
        earliest_loop_index = loop_index;
      }
    }

    // load keypoints, brief_descriptors
    std::string brief_path =
        config_.saved_pose_graph_dir + std::to_string(index) + "_briefdes.dat";
    std::ifstream brief_file(brief_path, std::ios::binary);
    std::string keypoints_path =
        config_.saved_pose_graph_dir + to_string(index) + "_keypoints.txt";
    FILE *keypoints_file;
    keypoints_file = fopen(keypoints_path.c_str(), "r");
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypoints_norm;
    std::vector<BRIEF::bitset> brief_descriptors;
    for (int i = 0; i < keypoints_num; i++)
    {
      BRIEF::bitset tmp_des;
      brief_file >> tmp_des;
      brief_descriptors.push_back(tmp_des);
      cv::KeyPoint tmp_keypoint;
      cv::KeyPoint tmp_keypoint_norm;
      double p_x, p_y, p_x_norm, p_y_norm;
      if (!fscanf(keypoints_file, "%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm,
                  &p_y_norm))
      {
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
        time_stamp, index, vio_translation, vio_rotation, pose_graph_translation,
        pose_graph_rotation, image, loop_index, loop_info, keypoints,
        keypoints_norm, brief_descriptors, config_.image_height,
        config_.image_width, config_.brief_pattern_file_path,
        config_.save_debug_image);
    KeyFrame *old_keyframe = nullptr;
    LoadKeyFrame(keyframe, old_keyframe, matched_2d_old_norm, matched_id);
    current_kf_attribute = keyframe->getAttributes();
    if (old_keyframe != nullptr)
    {
      old_kf_attribute = old_keyframe->getAttributes();
    }

    current_kf_thumb_image = keyframe->getThumbImage();

    return true;
}

void PoseGraph::UpdateKeyFrameLoop(
      int index, const Eigen::Matrix<double, 8, 1> &loop_info)
{
    auto keyframe = GetKeyFrame(index);
    if (!keyframe)
    {
      throw std::runtime_error("UpdateKeyFrameLoop(): keyframe not found");
    }

    keyframe->updateLoop(loop_info);
    if (config_.fast_relocalization &&
        (std::abs(loop_info(7)) < 30.0 &&
         Vector3d(loop_info(0), loop_info(1), loop_info(2)).norm() < 20.0))
    {
      auto old_keyframe = GetKeyFrame(keyframe->loop_index);
      Vector3d w_P_old, w_P_cur, vio_P_cur;
      Matrix3d w_R_old, w_R_cur, vio_R_cur;
      old_keyframe->getPose(w_P_old, w_R_old);
      keyframe->getVioPose(vio_P_cur, vio_R_cur);

      Vector3d relative_t;
      Quaterniond relative_q;
      relative_t = keyframe->getLoopRelativeT();
      relative_q = (keyframe->getLoopRelativeQ()).toRotationMatrix();
      w_P_cur = w_R_old * relative_t + w_P_old;
      w_R_cur = w_R_old * relative_q;
      double shift_yaw;
      Matrix3d shift_r;
      Vector3d shift_t;
      shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x();
      shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
      shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;

      {
        std::lock_guard<std::mutex> lock(drift_mutex_);
        drift_.yaw = shift_yaw;
        drift_.rotation = shift_r;
        drift_.translation = shift_t;
    }
  }
}

  PoseGraph::Pose PoseGraph::GetImuCameraPose() const
  {
    std::lock_guard<std::mutex> lock(imu_camera_pose_mutex_);
    return imu_camera_pose_;
  }

  int PoseGraph::GetCurrentSequenceCount() const
  {
    return current_sequence_count_;
  }

  PoseGraph::Drift PoseGraph::GetDrift() const { return drift_; }

  PoseGraph::Pose PoseGraph::GetWorldVio() const { return world_vio_; }

  KeyFrame::Attributes PoseGraph::GetKeyFrameAttribute(int index) const
  {
    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    auto it = std::find_if(keyframes_.begin(), keyframes_.end(),
                           [index](const std::shared_ptr<KeyFrame> &kf)
                           {
                             return kf->index == index;
                           });

    if (it != keyframes_.end())
      return (*it)->getAttributes();
    else
      return KeyFrame::Attributes();
  }

  std::vector<KeyFrame::Attributes> PoseGraph::GetKeyFrameAttributes() const
  {
    std::vector<KeyFrame::Attributes> attributes;
    std::lock_guard<std::mutex> lock(keyframes_mutex_);
    for (const auto &keyframe : keyframes_)
    {
      attributes.push_back(keyframe->getAttributes());
    }
    return attributes;
  }

  void PoseGraph::UpdateImuCameraPose(const Pose &imu_camera_pose)
  {
    std::lock_guard<std::mutex> lock(imu_camera_pose_mutex_);
    imu_camera_pose_ = imu_camera_pose;
  }

  void PoseGraph::StartOptimizationThread()
  {
    printf("Optimization thread started. \n");

    // Start optimization thread
    optimization_thread_ = std::thread([this]()
                                       {
    while (keep_running_) {
      // Perform optimization
      Optimize4DoF();
      auto kf_attributes = GetKeyFrameAttributes();
      if (event_observer_) {
        event_observer_->OnPoseGraphOptimization(kf_attributes);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    } });
  }
  namespace
  {
    template <typename T>
    T NormalizeAngle(const T &angle_degrees)
    {
      if (angle_degrees > T(180.0))
        return angle_degrees - T(360.0);
      else if (angle_degrees < T(-180.0))
        return angle_degrees + T(360.0);
      else
        return angle_degrees;
    };

    struct AngleManifoldFunctor
    {
      template <typename T>
      bool Plus(const T *theta_radians, const T *delta_theta_radians,
                T *theta_radians_plus_delta) const
      {
        *theta_radians_plus_delta =
            NormalizeAngle(*theta_radians + *delta_theta_radians);

        return true;
      }

      template <typename T>
      bool Minus(const T *theta_y_radians, const T *theta_x_radians,
                 T *theta_y_minus_x_radians) const
      {
        *theta_y_minus_x_radians =
            NormalizeAngle(*theta_y_radians - *theta_x_radians);

        return true;
      }
    };

    template <typename T>
    void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll,
                                      T R[9])
    {
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
    void RotationMatrixTranspose(const T R[9], T inv_R[9])
    {
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
    void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
    {
      r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
      r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
      r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
    };

    struct FourDOFError
    {
      FourDOFError(double t_x, double t_y, double t_z, double relative_yaw,
                   double pitch_i, double roll_i)
          : t_x(t_x),
            t_y(t_y),
            t_z(t_z),
            relative_yaw(relative_yaw),
            pitch_i(pitch_i),
            roll_i(roll_i) {}

      template <typename T>
      bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j,
                      const T *tj, T *residuals) const
      {
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

      static ceres::CostFunction *Create(const double t_x, const double t_y,
                                         const double t_z,
                                         const double relative_yaw,
                                         const double pitch_i,
                                         const double roll_i)
      {
        return (new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
            new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
      }

      double t_x, t_y, t_z;
      double relative_yaw, pitch_i, roll_i;
    };

    struct FourDOFWeightError
    {
      FourDOFWeightError(double t_x, double t_y, double t_z, double relative_yaw,
                         double pitch_i, double roll_i)
          : t_x(t_x),
            t_y(t_y),
            t_z(t_z),
            relative_yaw(relative_yaw),
            pitch_i(pitch_i),
            roll_i(roll_i)
      {
        weight = 1;
      }

      template <typename T>
      bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j,
                      const T *tj, T *residuals) const
      {
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

      static ceres::CostFunction *Create(const double t_x, const double t_y,
                                         const double t_z,
                                         const double relative_yaw,
                                         const double pitch_i,
                                         const double roll_i)
      {
        return (new ceres::AutoDiffCostFunction<FourDOFWeightError, 4, 1, 3, 1, 3>(
            new FourDOFWeightError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i)));
      }

      double t_x, t_y, t_z;
      double relative_yaw, pitch_i, roll_i;
      double weight;
    };
  } // namespace

} // namespace pose_graph
