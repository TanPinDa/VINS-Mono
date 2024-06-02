#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

class Estimator
{
public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double &timestamp);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();

    void GetLastestEstiamtedStates(Eigen::Vector3d &out_position,
                                   Eigen::Quaterniond &out_orientation,
                                   Eigen::Vector3d &out_linear_velocity,
                                   Eigen::Vector3d &out_imu_linear_acceleration_bias,
                                   Eigen::Vector3d &out_imu_angular_velocity_bias) const;

    void UpdateCameraImuTransform(Eigen::Vector3d *out_translation_camera_to_imu, Eigen::Matrix3d *out_rotation_camera_to_imu) const;

    void UpdateDriftCorrectionData(Eigen::Vector3d &out_drift_correct_translationMatrix3d, Eigen::Matrix3d &out_drift_correction_rotation) const;

    void UpdateKeyPoses(vector<Vector3d> out_key_poses) const;
    void UpdateCameraPoseInWorldFrame(Eigen::Vector3d &out_position, Eigen::Matrix3d &out_orientation) const;
    void UpdatePointClouds(std::vector<Eigen::Vector3d> &out_point_clouds) const;
    void UpdateMarginedPointClouds(std::vector<Eigen::Vector3d> &out_point_clouds) const;
    void UpdateKeyframePointClouds(std::vector<Eigen::Vector3d> &out_point_clouds,
                                   std::vector<std::vector<float>> &feature_2d_3d_matches) const;

    double GetTimestamp(const int &index) const;
    double GetImuCameraClockOffset() const;

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;

    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    // TODO consider using something like a queue or linked list

    MotionEstimator m_estimator;

    // relocalization variable
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];

    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;

    // Raw observations
    Vector3d linear_acceleration, angular_velocity;

    // State variables
    Vector3d gravity_;

    // Camera to IMU
    double imu_camera_clock_offset_;



private:
    bool InitialStructure();
    bool VisualInitialAlign();
    bool RelativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void SlideWindow();
    void SolveOdometry();
    void SlideWindowNew();
    void SlideWindowOld();
    void Optimization();
    void Vector2double();
    void Double2vector();
    bool FailureDetection();

    // Raw Observations
    vector<Vector3d> linear_acceleration_buffer_[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buffer_[(WINDOW_SIZE + 1)];
    double timestamp_[(WINDOW_SIZE + 1)];

    // Processed Obserrvations
    vector<double> dt_buffer_[(WINDOW_SIZE + 1)];
    IntegrationBase *pre_integrations_[(WINDOW_SIZE + 1)];

    // State variables
    Vector3d positions_[(WINDOW_SIZE + 1)];
    Matrix3d orientations_[(WINDOW_SIZE + 1)];
    Vector3d linear_velocities_[(WINDOW_SIZE + 1)];
    Vector3d imu_linear_acceleration_biases_[(WINDOW_SIZE + 1)];
    Vector3d imu_angular_velocity_biases_[(WINDOW_SIZE + 1)];

    // Camera to IMU
    Vector3d translation_cameras_to_imu_[NUM_OF_CAM];
    Matrix3d rotation_cameras_to_imu_[NUM_OF_CAM];

    // Relocalisation Corrections
    Matrix3d drift_correction_rotation_;
    Vector3d drift_correction_translation_;
    vector<Vector3d> key_poses_;

    // Unkown variables
    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    FeatureManager f_manager;
    InitialEXRotation initial_ex_rotation;
    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;

    double initial_timestamp;
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];
    double para_Retrive_Pose[SIZE_POSE];
    double para_Td[1][1];
    double para_Tr[1][1];
    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;
};
