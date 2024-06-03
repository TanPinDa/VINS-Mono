#include "estimator.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"

Estimator::Estimator() : f_manager{orientations_}
{
    spdlog::info("init begins");
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        translation_cameras_to_imu_[i] = TIC[i];
        rotation_cameras_to_imu_[i] = RIC[i];
    }
    f_manager.SetRotationCameraToImu(rotation_cameras_to_imu_);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    imu_camera_clock_offset_ = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        orientations_[i].setIdentity();
        positions_[i].setZero();
        linear_velocities_[i].setZero();
        imu_linear_acceleration_biases_[i].setZero();
        imu_angular_velocity_biases_[i].setZero();
        dt_buffer_[i].clear();
        linear_acceleration_buffer_[i].clear();
        angular_velocity_buffer_[i].clear();

        if (pre_integrations_[i] != nullptr)
            delete pre_integrations_[i];
        pre_integrations_[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        translation_cameras_to_imu_[i] = Vector3d::Zero();
        rotation_cameras_to_imu_[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    imu_camera_clock_offset_ = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correction_rotation_ = Matrix3d::Identity();
    drift_correction_translation_ = Vector3d::Zero();
}

void Estimator::processIMU(double dt, const Vector3d &input_linear_acceleration, const Vector3d &input_angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        linear_acceleration = input_linear_acceleration;
        angular_velocity = input_angular_velocity;
    }

    if (!pre_integrations_[frame_count])
    {
        pre_integrations_[frame_count] = new IntegrationBase{linear_acceleration, angular_velocity, imu_linear_acceleration_biases_[frame_count], imu_angular_velocity_biases_[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations_[frame_count]->push_back(dt, input_linear_acceleration, input_angular_velocity);
        // if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, input_linear_acceleration, input_angular_velocity);

        dt_buffer_[frame_count].push_back(dt);
        linear_acceleration_buffer_[frame_count].push_back(input_linear_acceleration);
        angular_velocity_buffer_[frame_count].push_back(input_angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = orientations_[j] * (linear_acceleration - imu_linear_acceleration_biases_[j]) - gravity_;
        Vector3d un_gyr = 0.5 * (angular_velocity + input_angular_velocity) - imu_angular_velocity_biases_[j];
        orientations_[j] *= Utility::deltaQuat(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = orientations_[j] * (input_linear_acceleration - imu_linear_acceleration_biases_[j]) - gravity_;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        positions_[j] += dt * linear_velocities_[j] + 0.5 * dt * dt * un_acc;
        linear_velocities_[j] += dt * un_acc;
    }
    linear_acceleration = input_linear_acceleration;
    angular_velocity = input_angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double &timestamp_sec)
{
    spdlog::debug("new image coming ------------------------------------------");
    spdlog::debug("Adding feature points {}", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, imu_camera_clock_offset_))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    spdlog::debug("this frame is--------------------{}", marginalization_flag ? "reject" : "accept");
    spdlog::debug(marginalization_flag ? "Non-keyframe" : "Keyframe");
    spdlog::debug("Solving {}", frame_count);
    spdlog::debug("number of feature: {}", f_manager.getFeatureCount());
    timestamp_[frame_count] = timestamp_sec;

    ImageFrame imageframe(image, timestamp_sec);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(timestamp_sec, imageframe));
    tmp_pre_integration = new IntegrationBase{linear_acceleration, angular_velocity, imu_linear_acceleration_biases_[frame_count], imu_angular_velocity_biases_[frame_count]};

    if (ESTIMATE_EXTRINSIC == 2)
    {
        spdlog::info("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations_[frame_count]->delta_q, calib_ric))
            {
                spdlog::warn("initial extrinsic rotation calib success");
                spdlog::warn("initial extrinsic rotation:\n {}", fmt::streamed(calib_ric));
                rotation_cameras_to_imu_[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if (ESTIMATE_EXTRINSIC != 2 && (timestamp_sec - initial_timestamp) > 0.1)
            {
                result = InitialStructure();
                initial_timestamp = timestamp_sec;
            }
            if (result)
            {
                solver_flag = NON_LINEAR;
                SolveOdometry();
                SlideWindow();
                f_manager.removeFailures();
                spdlog::info("Initialization finish!");
                last_R = orientations_[WINDOW_SIZE];
                last_P = positions_[WINDOW_SIZE];
                last_R0 = orientations_[0];
                last_P0 = positions_[0];
            }
            else
                SlideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        SolveOdometry();
        spdlog::debug("solver costs: {}ms", t_solve.toc());

        if (FailureDetection())
        {
            spdlog::warn("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            spdlog::warn("system reboot!");
            return;
        }

        TicToc t_margin;
        SlideWindow();
        f_manager.removeFailures();
        spdlog::debug("marginalization costs: {}ms", t_margin.toc());
        // prepare output of VINS
        key_poses_.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses_.push_back(positions_[i]);

        last_R = orientations_[WINDOW_SIZE];
        last_P = positions_[WINDOW_SIZE];
        last_R0 = orientations_[0];
        last_P0 = positions_[0];
    }
}
bool Estimator::InitialStructure()
{
    TicToc t_sfm;
    // check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        // spdlog::warn("IMU variation %f!", var);
        if (var < 0.25)
        {
            spdlog::info("IMU excitation not enouth!");
            // return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.matched_features_in_frames)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!RelativePose(relative_R, relative_T, l))
    {
        spdlog::info("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        spdlog::debug("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == timestamp_[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > timestamp_[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            spdlog::debug("Not enough points for solve pnp !");
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            spdlog::debug("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (VisualInitialAlign())
        return true;
    else
    {
        spdlog::info("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::VisualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    // solve scale
    bool result = VisualIMUAlignment(all_image_frame, imu_angular_velocity_biases_, gravity_, x);
    if (!result)
    {
        spdlog::debug("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[timestamp_[i]].R;
        Vector3d Pi = all_image_frame[timestamp_[i]].T;
        positions_[i] = Pi;
        orientations_[i] = Ri;
        all_image_frame[timestamp_[i]].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    rotation_cameras_to_imu_[0] = RIC[0];
    f_manager.SetRotationCameraToImu(rotation_cameras_to_imu_);
    f_manager.triangulate(positions_, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations_[i]->repropagate(Vector3d::Zero(), imu_angular_velocity_biases_[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        positions_[i] = s * positions_[i] - orientations_[i] * TIC[0] - (s * positions_[0] - orientations_[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            linear_velocities_[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.matched_features_in_frames.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(gravity_);
    double yaw = Utility::R2ypr(R0 * orientations_[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    gravity_ = R0 * gravity_;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        positions_[i] = rot_diff * positions_[i];
        orientations_[i] = rot_diff * orientations_[i];
        linear_velocities_[i] = rot_diff * linear_velocities_[i];
    }
    std::stringstream ss;
    ss << "g0     " << gravity_.transpose();
    spdlog::debug(ss.str());
    // spdlog::debug("g0    {}", fmt::streamed(g.transpose()));
    ss << "my R0   " << Utility::R2ypr(orientations_[0]).transpose();
    spdlog::debug(ss.str());

    return true;
}

bool Estimator::RelativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                spdlog::debug("average_parallax {0} choose l {1} and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::SolveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(positions_, translation_cameras_to_imu_, rotation_cameras_to_imu_);
        spdlog::debug("triangulation costs {}", t_tri.toc());
        Optimization();
    }
}

void Estimator::Vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = positions_[i].x();
        para_Pose[i][1] = positions_[i].y();
        para_Pose[i][2] = positions_[i].z();
        Quaterniond q{orientations_[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = linear_velocities_[i].x();
        para_SpeedBias[i][1] = linear_velocities_[i].y();
        para_SpeedBias[i][2] = linear_velocities_[i].z();

        para_SpeedBias[i][3] = imu_linear_acceleration_biases_[i].x();
        para_SpeedBias[i][4] = imu_linear_acceleration_biases_[i].y();
        para_SpeedBias[i][5] = imu_linear_acceleration_biases_[i].z();

        para_SpeedBias[i][6] = imu_angular_velocity_biases_[i].x();
        para_SpeedBias[i][7] = imu_angular_velocity_biases_[i].y();
        para_SpeedBias[i][8] = imu_angular_velocity_biases_[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = translation_cameras_to_imu_[i].x();
        para_Ex_Pose[i][1] = translation_cameras_to_imu_[i].y();
        para_Ex_Pose[i][2] = translation_cameras_to_imu_[i].z();
        Quaterniond q{rotation_cameras_to_imu_[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = imu_camera_clock_offset_;
}

void Estimator::Double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(orientations_[0]);
    Vector3d origin_P0 = positions_[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        spdlog::debug("euler singular point!");
        rot_diff = orientations_[0] * Quaterniond(para_Pose[0][6],
                                                  para_Pose[0][3],
                                                  para_Pose[0][4],
                                                  para_Pose[0][5])
                                          .toRotationMatrix()
                                          .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        orientations_[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        positions_[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                            para_Pose[i][1] - para_Pose[0][1],
                                            para_Pose[i][2] - para_Pose[0][2]) +
                        origin_P0;

        linear_velocities_[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                                    para_SpeedBias[i][1],
                                                    para_SpeedBias[i][2]);

        imu_linear_acceleration_biases_[i] = Vector3d(para_SpeedBias[i][3],
                                                      para_SpeedBias[i][4],
                                                      para_SpeedBias[i][5]);

        imu_angular_velocity_biases_[i] = Vector3d(para_SpeedBias[i][6],
                                                   para_SpeedBias[i][7],
                                                   para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        translation_cameras_to_imu_[i] = Vector3d(para_Ex_Pose[i][0],
                                                  para_Ex_Pose[i][1],
                                                  para_Ex_Pose[i][2]);
        rotation_cameras_to_imu_[i] = Quaterniond(para_Ex_Pose[i][6],
                                                  para_Ex_Pose[i][3],
                                                  para_Ex_Pose[i][4],
                                                  para_Ex_Pose[i][5])
                                          .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        imu_camera_clock_offset_ = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correction_rotation_ = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correction_translation_ = prev_relo_t - drift_correction_rotation_ * relo_t;
        relo_relative_t = relo_r.transpose() * (positions_[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * orientations_[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(orientations_[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        // cout << "vins relo " << endl;
        // cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        // cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
    }
}

bool Estimator::FailureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        spdlog::info(" little feature {}", f_manager.last_track_num);
        // return true;
    }
    if (imu_linear_acceleration_biases_[WINDOW_SIZE].norm() > 2.5)
    {
        spdlog::info(" big IMU acc bias estimation {}", imu_linear_acceleration_biases_[WINDOW_SIZE].norm());
        return true;
    }
    if (imu_angular_velocity_biases_[WINDOW_SIZE].norm() > 1.0)
    {
        spdlog::info(" big IMU gyr bias estimation {}", imu_angular_velocity_biases_[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        spdlog::info(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = positions_[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        spdlog::info(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        spdlog::info(" big z translation");
        return true;
    }
    Matrix3d tmp_R = orientations_[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        spdlog::info(" big delta_angle ");
        // return true;
    }
    return false;
}

void Estimator::Optimization()
{
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::Manifold *pose_manifold = new PoseManifold();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, pose_manifold);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::Manifold *pose_manifold = new PoseManifold();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, pose_manifold);
        if (!ESTIMATE_EXTRINSIC)
        {
            spdlog::debug("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            spdlog::debug("estimate extinsic param");
    }
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        // problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    Vector2double();

    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations_[j]->sum_dt > 10.0)
            continue;
        IMUFactor *imu_factor = new IMUFactor(pre_integrations_[j]);
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.matched_features_in_frames.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.matched_features_in_frames[0].point;

        for (auto &it_per_frame : it_per_id.matched_features_in_frames)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.matched_features_in_frames[0].pixel_velocity, it_per_frame.pixel_velocity,
                                                                  it_per_id.matched_features_in_frames[0].imu_camera_clock_offset_current_, it_per_frame.imu_camera_clock_offset_current_,
                                                                  it_per_id.matched_features_in_frames[0].pixel_coordinates.y(), it_per_frame.pixel_coordinates.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                /*
                double **para = new double *[5];
                para[0] = para_Pose[imu_i];
                para[1] = para_Pose[imu_j];
                para[2] = para_Ex_Pose[0];
                para[3] = para_Feature[feature_index];
                para[4] = para_Td[0];
                f_td->check(para);
                */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    spdlog::debug("visual measurement count: {}", f_m_cnt);
    spdlog::debug("prepare for ceres: {}", t_prepare.toc());

    if (relocalization_info)
    {
        // printf("set relocalization factor! \n");
        ceres::Manifold *pose_manifold = new PoseManifold();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, pose_manifold);
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.matched_features_in_frames.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if (start <= relo_frame_local_index)
            {
                while ((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                if ((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.matched_features_in_frames[0].point;

                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // cout << summary.BriefReport() << endl;
    spdlog::debug("Iterations : {}", static_cast<int>(summary.iterations.size()));
    spdlog::debug("solver costs: {}", t_solver.toc());

    Double2vector();

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        Vector2double();

        if (last_marginalization_info)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        {
            if (pre_integrations_[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations_[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.matched_features_in_frames.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.matched_features_in_frames[0].point;

                for (auto &it_per_frame : it_per_id.matched_features_in_frames)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.matched_features_in_frames[0].pixel_velocity, it_per_frame.pixel_velocity,
                                                                          it_per_id.matched_features_in_frames[0].imu_camera_clock_offset_current_, it_per_frame.imu_camera_clock_offset_current_,
                                                                          it_per_id.matched_features_in_frames[0].pixel_coordinates.y(), it_per_frame.pixel_coordinates.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        spdlog::debug("pre marginalization {} ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        spdlog::debug("marginalization {} ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            Vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            spdlog::debug("begin marginalization");
            marginalization_info->preMarginalize();
            spdlog::debug("end pre marginalization, {} ms", t_pre_margin.toc());

            TicToc t_margin;
            spdlog::debug("begin marginalization");
            marginalization_info->marginalize();
            spdlog::debug("end marginalization, {} ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    spdlog::debug("whole marginalization costs: {}", t_whole_marginalization.toc());

    spdlog::debug("whole time for ceres: {}", t_whole.toc());
}

void Estimator::SlideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = timestamp_[0];
        back_R0 = orientations_[0];
        back_P0 = positions_[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                orientations_[i].swap(orientations_[i + 1]);

                std::swap(pre_integrations_[i], pre_integrations_[i + 1]);

                dt_buffer_[i].swap(dt_buffer_[i + 1]);
                linear_acceleration_buffer_[i].swap(linear_acceleration_buffer_[i + 1]);
                angular_velocity_buffer_[i].swap(angular_velocity_buffer_[i + 1]);

                timestamp_[i] = timestamp_[i + 1];
                positions_[i].swap(positions_[i + 1]);
                linear_velocities_[i].swap(linear_velocities_[i + 1]);
                imu_linear_acceleration_biases_[i].swap(imu_linear_acceleration_biases_[i + 1]);
                imu_angular_velocity_biases_[i].swap(imu_angular_velocity_biases_[i + 1]);
            }
            timestamp_[WINDOW_SIZE] = timestamp_[WINDOW_SIZE - 1];
            positions_[WINDOW_SIZE] = positions_[WINDOW_SIZE - 1];
            linear_velocities_[WINDOW_SIZE] = linear_velocities_[WINDOW_SIZE - 1];
            orientations_[WINDOW_SIZE] = orientations_[WINDOW_SIZE - 1];
            imu_linear_acceleration_biases_[WINDOW_SIZE] = imu_linear_acceleration_biases_[WINDOW_SIZE - 1];
            imu_angular_velocity_biases_[WINDOW_SIZE] = imu_angular_velocity_biases_[WINDOW_SIZE - 1];

            delete pre_integrations_[WINDOW_SIZE];
            pre_integrations_[WINDOW_SIZE] = new IntegrationBase{linear_acceleration, angular_velocity, imu_linear_acceleration_biases_[WINDOW_SIZE], imu_angular_velocity_biases_[WINDOW_SIZE]};

            dt_buffer_[WINDOW_SIZE].clear();
            linear_acceleration_buffer_[WINDOW_SIZE].clear();
            angular_velocity_buffer_[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            SlideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buffer_[frame_count].size(); i++)
            {
                double tmp_dt = dt_buffer_[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buffer_[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buffer_[frame_count][i];

                pre_integrations_[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buffer_[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buffer_[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buffer_[frame_count - 1].push_back(tmp_angular_velocity);
            }

            timestamp_[frame_count - 1] = timestamp_[frame_count];
            positions_[frame_count - 1] = positions_[frame_count];
            linear_velocities_[frame_count - 1] = linear_velocities_[frame_count];
            orientations_[frame_count - 1] = orientations_[frame_count];
            imu_linear_acceleration_biases_[frame_count - 1] = imu_linear_acceleration_biases_[frame_count];
            imu_angular_velocity_biases_[frame_count - 1] = imu_angular_velocity_biases_[frame_count];

            delete pre_integrations_[WINDOW_SIZE];
            pre_integrations_[WINDOW_SIZE] = new IntegrationBase{linear_acceleration, angular_velocity, imu_linear_acceleration_biases_[WINDOW_SIZE], imu_angular_velocity_biases_[WINDOW_SIZE]};

            dt_buffer_[WINDOW_SIZE].clear();
            linear_acceleration_buffer_[WINDOW_SIZE].clear();
            angular_velocity_buffer_[WINDOW_SIZE].clear();

            SlideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::SlideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::SlideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * rotation_cameras_to_imu_[0];
        R1 = orientations_[0] * rotation_cameras_to_imu_[0];
        P0 = back_P0 + back_R0 * translation_cameras_to_imu_[0];
        P1 = positions_[0] + orientations_[0] * translation_cameras_to_imu_[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        if (relo_frame_stamp == timestamp_[i])
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

void Estimator::GetLastestEstiamtedStates(Eigen::Vector3d &out_position,
                                          Eigen::Quaterniond &out_orientation,
                                          Eigen::Vector3d &out_linear_velocity,
                                          Eigen::Vector3d &out_imu_linear_acceleration_bias,
                                          Eigen::Vector3d &out_imu_angular_velocity_bias) const
{
    out_position = positions_[WINDOW_SIZE];
    out_orientation = orientations_[WINDOW_SIZE]; // Implicit converion from Matrix to Quat
    out_linear_velocity = linear_velocities_[WINDOW_SIZE];
    out_imu_linear_acceleration_bias = imu_linear_acceleration_biases_[WINDOW_SIZE];
    out_imu_angular_velocity_bias = imu_angular_velocity_biases_[WINDOW_SIZE];
}

void Estimator::UpdateCameraImuTransform(Eigen::Vector3d *out_translation_camera_to_imu, Eigen::Matrix3d *out_rotation_camera_to_imu) const
{
    for (int i = 0; i < NUM_OF_CAM; ++i)
    {
        out_translation_camera_to_imu[i] = translation_cameras_to_imu_[i];
        out_rotation_camera_to_imu[i] = rotation_cameras_to_imu_[i];
    }
}

void Estimator::UpdateDriftCorrectionData(Eigen::Vector3d &out_drift_correction_translation, Eigen::Matrix3d &out_drift_correction_rotation) const
{
    out_drift_correction_translation = drift_correction_translation_;
    out_drift_correction_rotation = drift_correction_rotation_;
}
double Estimator::GetImuCameraClockOffset() const
{
    return imu_camera_clock_offset_;
}

void Estimator::UpdateKeyPoses(vector<Vector3d> out_key_poses) const
{

    out_key_poses = key_poses_;
}
void Estimator::UpdateCameraPoseInWorldFrame(Eigen::Vector3d &out_position, Eigen::Matrix3d &out_orientation) const
{
    // Update camera pose w.r.t world frame
    // NOTE: Overe here we use  latest imu data but the original code instead used the second latest. Unaware if this is intentional or just a typo
    out_position = positions_[WINDOW_SIZE] + orientations_[WINDOW_SIZE] * translation_cameras_to_imu_[0];

    // This there assumes that there is only one camera to publish, and it is the first one.
    out_orientation = orientations_[WINDOW_SIZE] * Quaterniond(rotation_cameras_to_imu_[0]);
}

void Estimator::UpdatePointClouds(std::vector<Eigen::Vector3d> &out_point_clouds) const
{
    for (auto &it_per_id : f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.matched_features_in_frames.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Eigen::Vector3d pts_i = it_per_id.matched_features_in_frames[0].point * it_per_id.estimated_depth;
        Eigen::Vector3d w_pts_i = orientations_[imu_i] * (rotation_cameras_to_imu_[0] * pts_i + translation_cameras_to_imu_[0]) + positions_[imu_i];
        out_point_clouds.push_back(w_pts_i);
    }
}

void Estimator::UpdateMarginedPointClouds(std::vector<Eigen::Vector3d> &out_point_clouds) const
{

    for (auto &it_per_id : f_manager.feature)
    {
        int used_num;
        used_num = it_per_id.matched_features_in_frames.size();
        if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        if (it_per_id.start_frame == 0 && it_per_id.matched_features_in_frames.size() <= 2 && it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.matched_features_in_frames[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = orientations_[imu_i] * (rotation_cameras_to_imu_[0] * pts_i + translation_cameras_to_imu_[0]) + positions_[imu_i];
            out_point_clouds.push_back(w_pts_i);
        }
    }
}

void Estimator::UpdateKeyframePointClouds(std::vector<Eigen::Vector3d> &out_point_clouds,
                                          std::vector<std::vector<float>> &feature_2d_3d_matches) const
{

    for (auto &it_per_id : f_manager.feature)
    {

        int frame_size = it_per_id.matched_features_in_frames.size();

        if (it_per_id.start_frame < WINDOW_SIZE - 2 && it_per_id.start_frame + frame_size - 1 >= WINDOW_SIZE - 2 && it_per_id.solve_flag == 1)
        {
            int imu_i = it_per_id.start_frame;
            int imu_j = WINDOW_SIZE - 2 - it_per_id.start_frame;

            Vector3d pts_i = it_per_id.matched_features_in_frames[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = orientations_[imu_i] * (rotation_cameras_to_imu_[0] * pts_i + translation_cameras_to_imu_[0]) + positions_[imu_i];
            out_point_clouds.push_back(w_pts_i);

            std::vector<float> feature_match;

            feature_match.push_back(it_per_id.matched_features_in_frames[imu_j].point.x());
            feature_match.push_back(it_per_id.matched_features_in_frames[imu_j].point.y());
            feature_match.push_back(it_per_id.matched_features_in_frames[imu_j].pixel_coordinates.x());
            feature_match.push_back(it_per_id.matched_features_in_frames[imu_j].pixel_coordinates.y());
            feature_match.push_back(it_per_id.feature_id);
            feature_2d_3d_matches.push_back(feature_match);
        }
    }
}

double Estimator::GetTimestamp(const int &index) const
{
    return timestamp_[index];
}