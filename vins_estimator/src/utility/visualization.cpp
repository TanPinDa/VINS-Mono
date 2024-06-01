#include "visualization.h"

using namespace ros;
using namespace Eigen;

EstimatorPublisher::EstimatorPublisher(ros::NodeHandle &n) : cameraposevisual(0, 1, 0, 1), keyframebasevisual(0.0, 0.0, 1.0, 1.0)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_relo_path = n.advertise<nav_msgs::Path>("relocalization_path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("history_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_relo_relative_pose = n.advertise<nav_msgs::Odometry>("relo_relative_pose", 1000);

    sum_of_path = 0;
    last_path = Vector3d(0.0, 0.0, 0.0);
    cameraposevisual.setScale(1);
    cameraposevisual.setLineWidth(0.05);
    keyframebasevisual.setScale(0.1);
    keyframebasevisual.setLineWidth(0.01);

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
    pub_latest_odometry.publish(odometry);
    sum_of_path = 0;
    last_path = Vector3d(0.0, 0.0, 0.0);
}

void EstimatorPublisher::printStatistics(const double &imu_camera_clock_offset, const Eigen::Vector3d translation_camera_to_imu[], const Matrix3d rotation_camera_to_imu[], const Vector3d &position, const Vector3d &linear_velocity, const double &compute_time)
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

    sum_of_path += (position - last_path).norm();
    last_path = position;
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (ESTIMATE_TD)
        ROS_INFO("td %f", imu_camera_clock_offset);
}
void EstimatorPublisher::PublishAll(const Estimator &estimator, const std_msgs::Header &header, const double &compute_time)
{

    estimator.GetLastestEstiamtedStates(position_estimated_current_,
                                        orientation_estimated_current_,
                                        linear_velocity_estimated_current_,
                                        imu_linear_acceleration_estimated_bias_,
                                        imu_angular_velocity_estimated_bias_);
    estimator.UpdateKeyPoses(key_poses_);
    imu_camera_clock_offset_ = estimator.GetImuCameraClockOffset();
    estimator.UpdateCameraImuTransform(translation_cameras_to_imu_, rotation_cameras_to_imu_);
    estimator.UpdateDriftCorrectionData(drift_correction_translation_, drift_correction_rotation_);

    // Update camera pose w.r.t world frame
    // NOTE: Overe here we use  latest imu data but the original code instead used the second latest. Unaware if this is intentional or just a typo
    camera_position_in_world_frame_ = position_estimated_current_ + orientation_estimated_current_ * translation_cameras_to_imu_[0];

    // This there assumes that there is only one camera to publish, and it is the first one.
    camera_orientation_in_world_frame_ = orientation_estimated_current_ * Quaterniond(rotation_cameras_to_imu_[0]);

    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
    {
        printStatistics(imu_camera_clock_offset_, translation_cameras_to_imu_, rotation_cameras_to_imu_, position_estimated_current_, linear_velocity_estimated_current_, compute_time);
        pubOdometry(position_estimated_current_, orientation_estimated_current_, linear_velocity_estimated_current_, drift_correction_translation_, drift_correction_rotation_, header);
        pubCameraPose(camera_position_in_world_frame_, Quaterniond(camera_orientation_in_world_frame_), header);
    }
    pubKeyPoses(key_poses_, header);

    pubPointCloud(estimator, header);
    pubTF(estimator, header);
    pubKeyframe(estimator);
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
void EstimatorPublisher::pubOdometry(const Vector3d &position, const Eigen::Quaterniond orientation, const Vector3d &linear_velocity,
                                     const Vector3d &drift_correction_translation, const Matrix3d &drift_correction_rotation, const std_msgs::Header &header)
{

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    UpdatePoseMessage(odometry.pose.pose, position, orientation);
    UpdateTwistMessage(odometry.twist.twist, linear_velocity);

    pub_odometry.publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path.header = header;
    path.header.frame_id = "world";
    path.poses.push_back(pose_stamped);
    pub_path.publish(path);

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
    relo_path.header = header;
    relo_path.header.frame_id = "world";
    relo_path.poses.push_back(pose_stamped);
    pub_relo_path.publish(relo_path);

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

void EstimatorPublisher::pubKeyPoses(const vector<Vector3d> &key_poses, const std_msgs::Header &header)
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
    pub_key_poses.publish(key_poses_msg_);
}

void EstimatorPublisher::pubCameraPose(const Vector3d &camera_position, const Eigen::Quaterniond &camera_orientation, const std_msgs::Header &header)
{
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    UpdatePoseMessage(odometry.pose.pose, camera_position, camera_orientation);
    pub_camera_pose.publish(odometry);
    cameraposevisual.reset();
    cameraposevisual.add_pose(camera_position, camera_orientation);
    cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
}
void EstimatorPublisher::pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.orientations_[imu_i] * (estimator.rotation_cameras_to_imu_[0] * pts_i + estimator.translation_cameras_to_imu_[0]) + estimator.positions_[imu_i];

        geometry_msgs::Point32 p;
        p.x = w_pts_i(0);
        p.y = w_pts_i(1);
        p.z = w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);

    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // if (it_per_id->start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //         continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 && it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.orientations_[imu_i] * (estimator.rotation_cameras_to_imu_[0] * pts_i + estimator.translation_cameras_to_imu_[0]) + estimator.positions_[imu_i];

            geometry_msgs::Point32 p;
            p.x = w_pts_i(0);
            p.y = w_pts_i(1);
            p.z = w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}

void EstimatorPublisher::pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag != Estimator::SolverFlag::NON_LINEAR)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.positions_[WINDOW_SIZE];
    correct_q = estimator.orientations_[WINDOW_SIZE];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.translation_cameras_to_imu_[0].x(),
                                    estimator.translation_cameras_to_imu_[0].y(),
                                    estimator.translation_cameras_to_imu_[0].z()));
    q.setW(Quaterniond(estimator.rotation_cameras_to_imu_[0]).w());
    q.setX(Quaterniond(estimator.rotation_cameras_to_imu_[0]).x());
    q.setY(Quaterniond(estimator.rotation_cameras_to_imu_[0]).y());
    q.setZ(Quaterniond(estimator.rotation_cameras_to_imu_[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.translation_cameras_to_imu_[0].x();
    odometry.pose.pose.position.y = estimator.translation_cameras_to_imu_[0].y();
    odometry.pose.pose.position.z = estimator.translation_cameras_to_imu_[0].z();
    Quaterniond tmp_q{estimator.rotation_cameras_to_imu_[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);
}

void EstimatorPublisher::pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
    {
        int i = WINDOW_SIZE - 2;
        // Vector3d P = estimator.positions_[i] + estimator.orientations_[i] * estimator.tic[0];
        Vector3d P = estimator.positions_[i];
        Quaterniond R = Quaterniond(estimator.orientations_[i]);

        nav_msgs::Odometry odometry;
        // what about sequence? Although can just set those to 0 usually
        odometry.header.stamp = ros::Time(estimator.Timestamps[WINDOW_SIZE - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        // printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose.publish(odometry);

        sensor_msgs::PointCloud point_cloud;
        // what about hedaer. Perhaps based on the full code we will know the header.
        point_cloud.header.stamp = ros::Time(estimator.Timestamps[WINDOW_SIZE - 2]);
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = it_per_id.feature_per_frame.size();
            if (it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.orientations_[imu_i] * (estimator.rotation_cameras_to_imu_[0] * pts_i + estimator.translation_cameras_to_imu_[0]) + estimator.positions_[imu_i];
                geometry_msgs::Point32 p;
                p.x = w_pts_i(0);
                p.y = w_pts_i(1);
                p.z = w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].point.y());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.x());
                p_2d.values.push_back(it_per_id.feature_per_frame[imu_j].uv.y());
                p_2d.values.push_back(it_per_id.feature_id);
                point_cloud.channels.push_back(p_2d);
            }
        }
        pub_keyframe_point.publish(point_cloud);
    }
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

    pub_relo_relative_pose.publish(odometry);
}