#include <stdio.h>

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include "estimator.h"
#include "parameters_ros.h"
#include "utility/visualization.h"

void updateCurrentOrientation(
    const Eigen::Vector3d &imu_angular_velocity_current,
    const Eigen::Vector3d &imu_angular_velocity_previous,
    const Eigen::Vector3d &imu_angular_velocity_bias,
    const Eigen::Quaterniond &previous_orientation, const double &dt,
    Eigen::Quaterniond &out_current_orientation) {
  Eigen::Vector3d ave_ang_vel =
      0.5 * (imu_angular_velocity_current + imu_angular_velocity_previous) -
      imu_angular_velocity_bias;
  Eigen::Vector3d delta_rot_vec = ave_ang_vel * dt;
  Eigen::Quaterniond delta_quat = Utility::deltaQuat(delta_rot_vec);
  out_current_orientation = previous_orientation * delta_quat;
}

void updateUnbiasedAccelerationInWorldFrame(
    const Eigen::Vector3d &imu_linear_acceleration,
    const Eigen::Vector3d &imu_linear_acceleration_bias,
    const Eigen::Quaterniond &orientation, const Eigen::Vector3d &gravity,
    Eigen::Vector3d &out_unbiased_accel_in_world_frame) {
  out_unbiased_accel_in_world_frame =
      orientation * (imu_linear_acceleration - imu_linear_acceleration_bias) -
      gravity;
}

void updateCurrentPositionAndVelocity(
    const Eigen::Vector3d &imu_linear_acceleration_previous,
    const Eigen::Vector3d &imu_linear_acceleration_current,
    const Eigen::Quaterniond &previous_orientation,
    const Eigen::Quaterniond &current_orientation,
    const Eigen::Vector3d &imu_linear_acceleration_bias,
    const Eigen::Vector3d &gravity, const double &dt,
    Eigen::Vector3d &out_position, Eigen::Vector3d &out_velocity) {
  Eigen::Vector3d prev_unbiased_accel;
  Eigen::Vector3d current_unbiased_accel;
  Eigen::Vector3d average_unbiased_accel;
  updateUnbiasedAccelerationInWorldFrame(
      imu_linear_acceleration_previous, imu_linear_acceleration_bias,
      previous_orientation, gravity, prev_unbiased_accel);
  updateUnbiasedAccelerationInWorldFrame(
      imu_linear_acceleration_current, imu_linear_acceleration_bias,
      current_orientation, gravity, current_unbiased_accel);
  average_unbiased_accel = 0.5 * (prev_unbiased_accel + current_unbiased_accel);
  out_position =
      out_position + out_velocity * dt + 0.5 * average_unbiased_accel * dt * dt;
  out_velocity = out_velocity + average_unbiased_accel * dt;
}

// thread: visual-inertial odometry

class EstimatorNode {
 public:
  EstimatorNode() {
    _nh = ros::NodeHandle("~");
    // _nh.setParam("config_file",
    // "/home/rosdev/workspace/ros_ws/src/VINS-Mono/config/euroc/euroc_config.yaml");
  }

  void Init() {
    readParameters(_nh);

    estimator_publisher = std::make_unique<EstimatorPublisher>(_nh);

    estimator_.setParameter();
    ROS_WARN("waiting for image and imu...");

    _sub_imu = _nh.subscribe("/imu0", 2000, &EstimatorNode::ImuCallback, this,
                             ros::TransportHints().tcpNoDelay());
    _sub_image = _nh.subscribe("/feature_tracker/feature", 2000,
                               &EstimatorNode::FeatureCallback, this);
    _sub_restart = _nh.subscribe("/feature_tracker/restart", 2000,
                                 &EstimatorNode::RestartCallback, this);
    _sub_relo_points =
        _nh.subscribe("/pose_graph/match_points", 2000,
                      &EstimatorNode::RelocalizationCallback, this);
    ROS_WARN("MADE SUBSCRIBERs");
  }

  void StartThread() {
    measurement_process = std::thread{std::bind(&EstimatorNode::Process, this)};
  }

 private:
  void Predict(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    if (init_imu) {
      latest_time = t;
      init_imu = false;
      return;
    }
    double dt = t - latest_time;
    latest_time = t;

    linear_acceleration_previous = linear_acceleration_current;
    angular_velocity_previous = angular_velocity_current;
    orientation_estimated_previous = orientation_estimated_current;

    linear_acceleration_current = {imu_msg->linear_acceleration.x,
                                   imu_msg->linear_acceleration.y,
                                   imu_msg->linear_acceleration.z};

    angular_velocity_current = {imu_msg->angular_velocity.x,
                                imu_msg->angular_velocity.y,
                                imu_msg->angular_velocity.z};

    updateCurrentOrientation(
        angular_velocity_previous, angular_velocity_current,
        imu_angular_velocity_estimated_bias, orientation_estimated_previous, dt,
        orientation_estimated_current);
    updateCurrentPositionAndVelocity(
        linear_acceleration_previous, linear_acceleration_current,
        orientation_estimated_previous, orientation_estimated_current,
        imu_linear_acceleration_estimated_bias, estimator_.g, dt,
        position_estimated_current, linear_velocity_estimated_current);
  }

  void Update() {
    TicToc t_predict;
    latest_time = current_time;
    position_estimated_current = estimator_.position[WINDOW_SIZE];
    orientation_estimated_current = estimator_.orientation[WINDOW_SIZE];
    linear_velocity_estimated_current = estimator_.linear_velocity[WINDOW_SIZE];
    imu_linear_acceleration_estimated_bias =
        estimator_.imu_linear_acceleration_bias[WINDOW_SIZE];
    imu_angular_velocity_estimated_bias =
        estimator_.imu_angular_velocity_bias[WINDOW_SIZE];
    linear_acceleration_current = estimator_.linear_acceleration;
    angular_velocity_current = estimator_.angular_velocity;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty();
         tmp_imu_buf.pop())
      Predict(tmp_imu_buf.front());
  }

  std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,
                        sensor_msgs::PointCloudConstPtr>>
  GetMeasurements() {
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,
                          sensor_msgs::PointCloudConstPtr>>
        measurements;

    while (true) {
      if (imu_buf.empty() || feature_buf.empty()) return measurements;

      if (!(imu_buf.back()->header.stamp.toSec() >
            feature_buf.front()->header.stamp.toSec() + estimator_.td)) {
        // ROS_WARN("wait for imu, only should happen at the beginning");
        sum_of_wait++;
        return measurements;
      }

      if (!(imu_buf.front()->header.stamp.toSec() <
            feature_buf.front()->header.stamp.toSec() + estimator_.td)) {
        ROS_WARN("throw img, only should happen at the beginning");
        feature_buf.pop();
        continue;
      }
      sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
      feature_buf.pop();

      std::vector<sensor_msgs::ImuConstPtr> IMUs;
      while (imu_buf.front()->header.stamp.toSec() <
             img_msg->header.stamp.toSec() + estimator_.td) {
        IMUs.emplace_back(imu_buf.front());
        imu_buf.pop();
      }
      IMUs.emplace_back(imu_buf.front());
      if (IMUs.empty()) ROS_WARN("no imu between two image");
      measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
  }

  void ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg) {
    if (imu_msg->header.stamp.toSec() <= last_imu_t) {
      ROS_WARN("imu message in disorder!");
      return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
      std::lock_guard<std::mutex> lg(m_state);
      Predict(imu_msg);
      std_msgs::Header header = imu_msg->header;
      header.frame_id = "world";
      if (estimator_.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        estimator_publisher->pubLatestOdometry(
            position_estimated_current, orientation_estimated_previous,
            linear_velocity_estimated_current, header);
    }
  }

  void FeatureCallback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
    if (!init_feature) {
      // skip the first detected feature, which doesn't contain optical flow
      // speed
      init_feature = 1;
      return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
  }

  void RestartCallback(const std_msgs::BoolConstPtr &restart_msg) {
    if (restart_msg->data == true) {
      ROS_WARN("restart the estimator!");
      m_buf.lock();
      while (!feature_buf.empty()) feature_buf.pop();
      while (!imu_buf.empty()) imu_buf.pop();
      m_buf.unlock();
      m_estimator.lock();
      estimator_.clearState();
      estimator_.setParameter();
      m_estimator.unlock();
      current_time = -1;
      last_imu_t = 0;
    }
    return;
  }

  void RelocalizationCallback(
      const sensor_msgs::PointCloudConstPtr &points_msg) {
    // printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
  }

  // thread: visual-inertial odometry
  void Process() {
    while (true) {
      std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>,
                            sensor_msgs::PointCloudConstPtr>>
          measurements;
      std::unique_lock<std::mutex> lk(m_buf);
      con.wait(lk,
               [&] { return (measurements = GetMeasurements()).size() != 0; });
      lk.unlock();
      m_estimator.lock();
      for (auto &measurement : measurements) {
        auto img_msg = measurement.second;
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        for (auto &imu_msg : measurement.first) {
          double t = imu_msg->header.stamp.toSec();
          double img_t = img_msg->header.stamp.toSec() + estimator_.td;
          if (t <= img_t) {
            if (current_time < 0) current_time = t;
            double dt = t - current_time;
            ROS_ASSERT(dt >= 0);
            current_time = t;
            dx = imu_msg->linear_acceleration.x;
            dy = imu_msg->linear_acceleration.y;
            dz = imu_msg->linear_acceleration.z;
            rx = imu_msg->angular_velocity.x;
            ry = imu_msg->angular_velocity.y;
            rz = imu_msg->angular_velocity.z;
            estimator_.processIMU(dt, Vector3d(dx, dy, dz),
                                  Vector3d(rx, ry, rz));
            // printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx,
            // ry, rz);
          } else {
            double dt_1 = img_t - current_time;
            double dt_2 = t - img_t;
            current_time = img_t;
            ROS_ASSERT(dt_1 >= 0);
            ROS_ASSERT(dt_2 >= 0);
            ROS_ASSERT(dt_1 + dt_2 > 0);
            double w1 = dt_2 / (dt_1 + dt_2);
            double w2 = dt_1 / (dt_1 + dt_2);
            dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
            rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
            ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
            rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
            estimator_.processIMU(dt_1, Vector3d(dx, dy, dz),
                                  Vector3d(rx, ry, rz));
            // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz,
            // rx, ry, rz);
          }
        }
        // set relocalization frame
        sensor_msgs::PointCloudConstPtr relo_msg = NULL;
        while (!relo_buf.empty()) {
          relo_msg = relo_buf.front();
          relo_buf.pop();
        }
        if (relo_msg != NULL) {
          vector<Vector3d> match_points;
          double frame_stamp = relo_msg->header.stamp.toSec();
          for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
            Vector3d u_v_id;
            u_v_id.x() = relo_msg->points[i].x;
            u_v_id.y() = relo_msg->points[i].y;
            u_v_id.z() = relo_msg->points[i].z;
            match_points.push_back(u_v_id);
          }
          Vector3d relo_t(relo_msg->channels[0].values[0],
                          relo_msg->channels[0].values[1],
                          relo_msg->channels[0].values[2]);
          Quaterniond relo_q(
              relo_msg->channels[0].values[3], relo_msg->channels[0].values[4],
              relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
          Matrix3d relo_r = relo_q.toRotationMatrix();
          int frame_index;
          frame_index = relo_msg->channels[0].values[7];
          estimator_.setReloFrame(frame_stamp, frame_index, match_points,
                                  relo_t, relo_r);
        }

        ROS_DEBUG("processing vision data with stamp %f \n",
                  img_msg->header.stamp.toSec());

        TicToc t_s;
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
        for (unsigned int i = 0; i < img_msg->points.size(); i++) {
          int v = img_msg->channels[0].values[i] + 0.5;
          int feature_id = v / NUM_OF_CAM;
          int camera_id = v % NUM_OF_CAM;
          double x = img_msg->points[i].x;
          double y = img_msg->points[i].y;
          double z = img_msg->points[i].z;
          double p_u = img_msg->channels[1].values[i];
          double p_v = img_msg->channels[2].values[i];
          double velocity_x = img_msg->channels[3].values[i];
          double velocity_y = img_msg->channels[4].values[i];
          ROS_ASSERT(z == 1);
          Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
          xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
          image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }
        estimator_.processImage(image, img_msg->header.stamp.toSec());

        double whole_t = t_s.toc();
        estimator_publisher->printStatistics(estimator_, whole_t);
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world";

        estimator_publisher->pubOdometry(estimator_, header);
        estimator_publisher->pubKeyPoses(estimator_, header);
        estimator_publisher->pubCameraPose(estimator_, header);
        estimator_publisher->pubPointCloud(estimator_, header);
        estimator_publisher->pubTF(estimator_, header);
        estimator_publisher->pubKeyframe(estimator_);
        if (relo_msg != NULL)
          estimator_publisher->pubRelocalization(estimator_);
        // ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(),
        // ros::Time::now().toSec());
      }
      m_estimator.unlock();
      m_buf.lock();
      m_state.lock();
      if (estimator_.solver_flag == Estimator::SolverFlag::NON_LINEAR) Update();
      m_state.unlock();
      m_buf.unlock();
    }
  }
  ros::NodeHandle _nh;

  std::condition_variable con;
  double current_time = -1;
  queue<sensor_msgs::ImuConstPtr> imu_buf;
  queue<sensor_msgs::PointCloudConstPtr> feature_buf;
  queue<sensor_msgs::PointCloudConstPtr> relo_buf;
  int sum_of_wait = 0;

  std::mutex m_buf;
  std::mutex m_state;
  std::mutex i_buf;
  std::mutex m_estimator;

  double latest_time;
  Eigen::Vector3d position_estimated_current;
  Eigen::Vector3d linear_velocity_estimated_current;
  Eigen::Vector3d imu_linear_acceleration_estimated_bias;
  Eigen::Vector3d imu_angular_velocity_estimated_bias;
  Eigen::Quaterniond orientation_estimated_current;
  Eigen::Vector3d linear_acceleration_current;
  Eigen::Vector3d angular_velocity_current;

  Eigen::Quaterniond orientation_estimated_previous;
  Eigen::Vector3d linear_acceleration_previous;
  Eigen::Vector3d angular_velocity_previous;

  bool init_feature = 0;
  bool init_imu = 1;
  double last_imu_t = 0;
  // Will be a class varaible
  std::unique_ptr<EstimatorPublisher> estimator_publisher;
  std::thread measurement_process;
  Estimator estimator_;
  ros::Subscriber _sub_imu;
  ros::Subscriber _sub_image;
  ros::Subscriber _sub_restart;
  ros::Subscriber _sub_relo_points;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "vins_estimator");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME,
                                 ros::console::levels::Info);
  EstimatorNode estimator_node;
  estimator_node.Init();
  estimator_node.StartThread();

#ifdef EIGEN_DONT_PARALLELIZE
  ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

  ros::spin();

  return 0;
}
