#include "visualization.h"

using namespace ros;
using namespace Eigen;

EstimatorPublisher::EstimatorPublisher(ros::NodeHandle &n) : caemra_pose_visualization_(0, 1, 0, 1), keyframe_base_visualization_(0.0, 0.0, 1.0, 1.0)
{
    pub_latest_odometry_ = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path_ = n.advertise<nav_msgs::Path>("path", 1000);
    pub_relo_path_ = n.advertise<nav_msgs::Path>("relocalization_path", 1000);
    pub_odometry_ = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud_ = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud_ = n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
    pub_key_poses_ = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose_ = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual_ = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose_ = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point_ = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic_ = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_relo_relative_pose_ = n.advertise<nav_msgs::Odometry>("relo_relative_pose", 1000);

    sum_of_path_ = 0;
    last_path_ = Vector3d(0.0, 0.0, 0.0);
    caemra_pose_visualization_.setScale(1);
    caemra_pose_visualization_.setLineWidth(0.05);
    keyframe_base_visualization_.setScale(0.1);
    keyframe_base_visualization_.setLineWidth(0.01);

    key_poses_msg_.header.frame_id = "world";
    key_poses_msg_.lifetime = ros::Duration();

    key_poses_msg_.ns = "key_poses";
    key_poses_msg_.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses_msg_.action = visualization_msgs::Marker::ADD;
    key_poses_msg_.pose.orientation.w = 1.0;

    // static int key_poses_id = 0;
    key_poses_msg_.id = 0; // key_poses_id++;
    key_poses_msg_.scale.x = 0.05;
    key_poses_msg_.scale.y = 0.05;
    key_poses_msg_.scale.z = 0.05;
    key_poses_msg_.color.r = 1.0;
    key_poses_msg_.color.a = 1.0;
}

void EstimatorPublisher::pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header)
{
    Eigen::Quaterniond quadrotor_Q = Q;

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry_.publish(odometry);
    sum_of_path_ = 0;
    last_path_ = Vector3d(0.0, 0.0, 0.0);
}

void EstimatorPublisher::PrintStatistics(const double &imu_camera_clock_offset, const Eigen::Vector3d translation_camera_to_imu[], const Matrix3d rotation_camera_to_imu[], const Vector3d &position, const Vector3d &linear_velocity, const double &compute_time)
{
    printf("position: %f, %f, %f\r", position.x(), position.y(), position.z());
    ROS_DEBUG_STREAM("position: " << position.transpose());
    ROS_DEBUG_STREAM("orientation: " << linear_velocity.transpose());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // ROS_DEBUG("calibration result for camera %d", i);
        ROS_DEBUG_STREAM("extirnsic tic: " << translation_camera_to_imu[i].transpose());
        ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(rotation_camera_to_imu[i]).transpose());
        if (ESTIMATE_EXTRINSIC)
        {
            cv::FileStorage fs(EX_CALIB_RESULT_PATH, cv::FileStorage::WRITE);
            cv::Mat cv_R, cv_T;
            cv::eigen2cv(rotation_camera_to_imu[i], cv_R);
            cv::eigen2cv(translation_camera_to_imu[i], cv_T);
            fs << "extrinsicRotation" << cv_R << "extrinsicTranslation" << cv_T;
            fs.release();
        }
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += compute_time;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", compute_time);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path_ += (position - last_path_).norm();
    last_path_ = position;
    ROS_DEBUG("sum of path %f", sum_of_path_);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", imu_camera_clock_offset);
}
void EstimatorPublisher::PublishAll(const Estimator &estimator, const std_msgs::Header &header, const double &compute_time)
{

    std::vector<Eigen::Vector3d> point_clouds;
    std::vector<Eigen::Vector3d> marginalised_point_clouds;
    std::vector<Eigen::Vector3d> keyframe_point_clouds;
    std::vector<std::vector<float>> keyframe_feature_pairs;

    estimator.GetLastestEstiamtedStates(position_estimated_current_,
                                        orientation_estimated_current_,
                                        linear_velocity_estimated_current_,
                                        imu_linear_acceleration_estimated_bias_,
                                        imu_angular_velocity_estimated_bias_);
    estimator.UpdateKeyPoses(key_poses_);
    imu_camera_clock_offset_ = estimator.GetImuCameraClockOffset();
    estimator.UpdateCameraImuTransform(translation_cameras_to_imu_, rotation_cameras_to_imu_);
    estimator.UpdateDriftCorrectionData(drift_correction_translation_, drift_correction_rotation_);
    estimator.UpdatePointClouds(point_clouds);
    estimator.UpdateMarginedPointClouds(marginalised_point_clouds);
    estimator.UpdateKeyframePointClouds(keyframe_point_clouds, keyframe_feature_pairs);

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        PrintStatistics(imu_camera_clock_offset_, translation_cameras_to_imu_, rotation_cameras_to_imu_, position_estimated_current_, linear_velocity_estimated_current_, compute_time);
        PubOdometry(position_estimated_current_, orientation_estimated_current_, linear_velocity_estimated_current_, drift_correction_translation_, drift_correction_rotation_, header);
        PubCameraPose(camera_position_in_world_frame_, Quaterniond(camera_orientation_in_world_frame_), header);
        PubPointCloud(point_clouds, marginalised_point_clouds, header);
        PubTF(position_estimated_current_,
              orientation_estimated_current_, translation_cameras_to_imu_[0], Quaterniond(rotation_cameras_to_imu_[0]), header);
        if (estimator.marginalization_flag == 0)
        {
            PubKeyframe(position_estimated_current_,
                        orientation_estimated_current_, keyframe_point_clouds, keyframe_feature_pairs, estimator.GetTimestamp(WINDOW_SIZE - 2));
        }
    }
    PubKeyPoses(key_poses_, header);
}
void EstimatorPublisher::UpdatePoseMessage(geometry_msgs::Pose &pose_msg, const Vector3d &position, const Eigen::Quaterniond &orientation)
{
    pose_msg.position.x = position.x();
    pose_msg.position.y = position.y();
    pose_msg.position.z = position.z();
    pose_msg.orientation.x = orientation.x();
    pose_msg.orientation.y = orientation.y();
    pose_msg.orientation.z = orientation.z();
    pose_msg.orientation.w = orientation.w();
}

void EstimatorPublisher::UpdateTwistMessage(geometry_msgs::Twist twist_msg, const Eigen::Vector3d &velocity)
{
    twist_msg.linear.x = velocity.x();
    twist_msg.linear.y = velocity.y();
    twist_msg.linear.z = velocity.z();
}
void EstimatorPublisher::PubOdometry(const Vector3d &position, const Eigen::Quaterniond &orientation, const Vector3d &linear_velocity,
                                     const Vector3d &drift_correction_translation, const Matrix3d &drift_correction_rotation, const std_msgs::Header &header)
{

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    UpdatePoseMessage(odometry.pose.pose, position, orientation);
    UpdateTwistMessage(odometry.twist.twist, linear_velocity);

    pub_odometry_.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path_msg_.header = header;
    path_msg_.header.frame_id = "world";
    path_msg_.poses.push_back(pose_stamped);
    pub_path_.publish(path_msg_);

    Vector3d correct_t;
    Vector3d correct_v;
    Quaterniond correct_q;
    correct_t = drift_correction_rotation * position + drift_correction_translation;
    correct_q = drift_correction_rotation * orientation;
    odometry.pose.pose.position.x = correct_t.x();
    odometry.pose.pose.position.y = correct_t.y();
    odometry.pose.pose.position.z = correct_t.z();
    odometry.pose.pose.orientation.x = correct_q.x();
    odometry.pose.pose.orientation.y = correct_q.y();
    odometry.pose.pose.orientation.z = correct_q.z();
    odometry.pose.pose.orientation.w = correct_q.w();

    pose_stamped.pose = odometry.pose.pose;
    relo_path_msg_.header = header;
    relo_path_msg_.header.frame_id = "world";
    relo_path_msg_.poses.push_back(pose_stamped);
    pub_relo_path_.publish(relo_path_msg_);

    // write result to file
    ofstream foutC(VINS_RESULT_PATH, ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    foutC.precision(0);
    foutC << header.stamp.toSec() * 1e9 << ",";
    foutC.precision(5);
    foutC << position.x() << ","
          << position.y() << ","
          << position.z() << ","
          << orientation.w() << ","
          << orientation.x() << ","
          << orientation.y() << ","
          << orientation.z() << ","
          << linear_velocity.x() << ","
          << linear_velocity.y() << ","
          << linear_velocity.z() << "," << endl;
    foutC.close();
}

void EstimatorPublisher::PubKeyPoses(const vector<Vector3d> &key_poses, const std_msgs::Header &header)
{
    if (key_poses.size() == 0)
        return;

    key_poses_msg_.header = header;
    key_poses_msg_.header.frame_id = "world";

    key_poses_msg_.lifetime = ros::Duration();

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses_msg_.points.push_back(pose_marker);
    }
    pub_key_poses_.publish(key_poses_msg_);
}

void EstimatorPublisher::PubCameraPose(const Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, const std_msgs::Header &header)
{
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    UpdatePoseMessage(odometry.pose.pose, camera_position, camera_orientation);
    pub_camera_pose_.publish(odometry);
    caemra_pose_visualization_.reset();
    caemra_pose_visualization_.add_pose(camera_position, camera_orientation);
    caemra_pose_visualization_.publish_by(pub_camera_pose_visual_, odometry.header);
}
void EstimatorPublisher::PubPointCloud(const std::vector<Eigen::Vector3d> &point_clouds, const std::vector<Eigen::Vector3d> &margined_point_clouds, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    for (Eigen::Vector3d point : point_clouds)
    {
        geometry_msgs::Point32 p;
        p.x = point.x();
        p.y = point.y();
        p.z = point.z();
        point_cloud.points.push_back(p);
    }
    pub_point_cloud_.publish(point_cloud);

    // pub margined points
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;
    for (Eigen::Vector3d point : margined_point_clouds)
    {
        geometry_msgs::Point32 p;
        p.x = point.x();
        p.y = point.y();
        p.z = point.z();
        margin_cloud.points.push_back(p);
    }
    pub_margin_cloud_.publish(margin_cloud);
}

void EstimatorPublisher::PubTF(const Eigen::Vector3d &position,
                               const Eigen::Quaterniond &orientation,
                               const Eigen::Vector3d &translation_camera_to_imu,
                               const Eigen::Quaterniond &rotation_camera_to_imu,
                               const std_msgs::Header &header)
{

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame

    transform.setOrigin(tf::Vector3(position.x(),
                                    position.y(),
                                    position.z()));
    q.setW(orientation.w());
    q.setX(orientation.x());
    q.setY(orientation.y());
    q.setZ(orientation.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(translation_camera_to_imu.x(),
                                    translation_camera_to_imu.y(),
                                    translation_camera_to_imu.z()));
    q.setW(rotation_camera_to_imu.w());
    q.setX(rotation_camera_to_imu.x());
    q.setY(rotation_camera_to_imu.y());
    q.setZ(rotation_camera_to_imu.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    UpdatePoseMessage(odometry.pose.pose, translation_camera_to_imu, rotation_camera_to_imu);
    pub_extrinsic_.publish(odometry);
}

void EstimatorPublisher::PubKeyframe(
    const Eigen::Vector3d &position,
    const Eigen::Quaterniond &orientation,
    const std::vector<Eigen::Vector3d> &point_clouds,
    std::vector<std::vector<float>> &feature_2d_3d_matches,
    const double &timestamp_2_back)
{
    // pub camera pose, 2D-3D points of keyframe
    int i = WINDOW_SIZE - 2;
    // Should it be publishing pos and quat of imu or camera....
    // Vector3d P = estimator.positions_[i] + estimator.orientations_[i] * estimator.tic[0];

    nav_msgs::Odometry odometry;
    // what about sequence? Although can just set those to 0 usually
    odometry.header.stamp = ros::Time(timestamp_2_back);
    odometry.header.frame_id = "world";
    UpdatePoseMessage(odometry.pose.pose,position,orientation);
    pub_keyframe_pose_.publish(odometry);


    // Pub pointclouds
    sensor_msgs::PointCloud point_cloud;
    // what about hedaer. Perhaps based on the full code we will know the header.
    point_cloud.header.stamp = ros::Time(timestamp_2_back);
    point_cloud.header.frame_id = "world";

    for (Eigen::Vector3d point : point_clouds)
    {
        geometry_msgs::Point32 p;
        p.x = point.x();
        p.y = point.y();
        p.z = point.z();
        point_cloud.points.push_back(p);
    }

    for (std::vector<float> feature_2d_3d_match : feature_2d_3d_matches)
    {
        sensor_msgs::ChannelFloat32 p_2d;
        p_2d.values = feature_2d_3d_match;
        point_cloud.channels.push_back(p_2d);
    }

    pub_keyframe_point_.publish(point_cloud);
}

void EstimatorPublisher::pubRelocalization(const Estimator &estimator)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(estimator.relo_frame_stamp);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.relo_relative_t.x();
    odometry.pose.pose.position.y = estimator.relo_relative_t.y();
    odometry.pose.pose.position.z = estimator.relo_relative_t.z();
    odometry.pose.pose.orientation.x = estimator.relo_relative_q.x();
    odometry.pose.pose.orientation.y = estimator.relo_relative_q.y();
    odometry.pose.pose.orientation.z = estimator.relo_relative_q.z();
    odometry.pose.pose.orientation.w = estimator.relo_relative_q.w();
    odometry.twist.twist.linear.x = estimator.relo_relative_yaw;
    odometry.twist.twist.linear.y = estimator.relo_frame_index;

    pub_relo_relative_pose_.publish(odometry);
}