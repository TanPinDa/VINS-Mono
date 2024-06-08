#include "feature_manager.h"
#include "spdlog/spdlog.h"

int FeatureOccurrencesAcrossFrames::endFrame()
{
    return start_frame + matched_features_in_frames.size() - 1;
}

FeatureManager::FeatureManager()

{
    for (int i = 0; i < NUM_OF_CAM; i++)
        rotation_of_cameras_to_imu_[i].setIdentity();
}

void FeatureManager::SetRotationCameraToImu(Matrix3d rotation_of_cameras_to_imu[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        rotation_of_cameras_to_imu_[i] = rotation_of_cameras_to_imu[i];
    }
}

void FeatureManager::clearState()
{
    feature_tracks.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (FeatureOccurrencesAcrossFrames &feature_track : feature_tracks)
    {

        feature_track.used_num = feature_track.matched_features_in_frames.size();

        if (feature_track.used_num >= 2 && feature_track.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, FeatureBase>>> &image, double td)
{
    spdlog::debug("input feature: {}", (int)image.size());
    spdlog::debug("num of feature: {}", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts : image)
    {
        FeatureObservation f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        auto it = find_if(feature_tracks.begin(), feature_tracks.end(), [feature_id](const FeatureOccurrencesAcrossFrames &it)
                          {
            return it.feature_id == feature_id;
                          });

        if (it == feature_tracks.end())
        {
            feature_tracks.push_back(FeatureOccurrencesAcrossFrames(feature_id, frame_count));
            feature_tracks.back().matched_features_in_frames.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id)
        {
            it->matched_features_in_frames.push_back(f_per_fra);
            last_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &feature_track : feature_tracks)
    {
        if (feature_track.start_frame <= frame_count - 2 &&
            feature_track.start_frame + int(feature_track.matched_features_in_frames.size()) - 1 >= frame_count - 1)
        {
            parallax_sum += compensatedParallax2(feature_track, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        spdlog::debug("parallax_sum: {0}, parallax_num: {1}", parallax_sum, parallax_num);
        spdlog::debug("current parallax: {0}", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    spdlog::debug("debug show");
    for (auto &feature_track : feature_tracks)
    {
        assert(feature_track.matched_features_in_frames.size() != 0);
        assert(feature_track.start_frame >= 0);
        assert(feature_track.used_num >= 0);

        spdlog::debug("{0},{1},{2} ", feature_track.feature_id, feature_track.used_num, feature_track.start_frame);
        int sum = 0;
        for (auto &j : feature_track.matched_features_in_frames)
        {
            spdlog::debug("{},", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        assert(feature_track.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &feature_track : feature_tracks)
    {
        if (feature_track.start_frame <= frame_count_l && feature_track.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - feature_track.start_frame;
            int idx_r = frame_count_r - feature_track.start_frame;

            a = feature_track.matched_features_in_frames[idx_l].point;

            b = feature_track.matched_features_in_frames[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &feature_track : feature_tracks)
    {
        feature_track.used_num = feature_track.matched_features_in_frames.size();
        if (!(feature_track.used_num >= 2 && feature_track.start_frame < WINDOW_SIZE - 2))
            continue;

        feature_track.estimated_depth = 1.0 / x(++feature_index);
        //spdlog::info("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (feature_track.estimated_depth < 0)
        {
            feature_track.solve_flag = 2;
        }
        else
            feature_track.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature_tracks.begin(), it_next = feature_tracks.begin();
         it != feature_tracks.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature_tracks.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &feature_track : feature_tracks)
    {
        feature_track.used_num = feature_track.matched_features_in_frames.size();
        if (!(feature_track.used_num >= 2 && feature_track.start_frame < WINDOW_SIZE - 2))
            continue;
        feature_track.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &feature_track : feature_tracks)
    {
        feature_track.used_num = feature_track.matched_features_in_frames.size();
        if (!(feature_track.used_num >= 2 && feature_track.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / feature_track.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

void FeatureManager::triangulate(const Matrix3d imu_orientations_wrt_world[], const Vector3d translations_imu_to_world[], const Vector3d translations_camera_to_imu[], const Matrix3d rotations_camera_to_imu[])
{
    for (auto &feature_track : feature_tracks)
    {
        feature_track.used_num = feature_track.matched_features_in_frames.size();
        if (!(feature_track.used_num >= 2 && feature_track.start_frame < WINDOW_SIZE - 2))
            continue;

        if (feature_track.estimated_depth > 0)
            continue;
        int imu_i = feature_track.start_frame, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * feature_track.matched_features_in_frames.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = translations_imu_to_world[imu_i] + imu_orientations_wrt_world[imu_i] * translations_camera_to_imu[0];
        Eigen::Matrix3d R0 = imu_orientations_wrt_world[imu_i] * rotations_camera_to_imu[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : feature_track.matched_features_in_frames)
        {
            imu_j++;

            Eigen::Vector3d t1 = translations_imu_to_world[imu_j] + imu_orientations_wrt_world[imu_j] * translations_camera_to_imu[0];
            Eigen::Matrix3d R1 = imu_orientations_wrt_world[imu_j] * rotation_of_cameras_to_imu_[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        feature_track.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (feature_track.estimated_depth < 0.1)
        {
            feature_track.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{   
    // This function does not seem to be used anywhere.
    throw std::runtime_error("Functionality no implemented.");
    int i = -1;
    for (auto it = feature_tracks.begin(), it_next = feature_tracks.begin();
         it != feature_tracks.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature_tracks.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature_tracks.begin(), it_next = feature_tracks.begin();
         it != feature_tracks.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->matched_features_in_frames[0].point;  
            it->matched_features_in_frames.erase(it->matched_features_in_frames.begin());
            if (it->matched_features_in_frames.size() < 2)
            {
                feature_tracks.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature_tracks.begin(), it_next = feature_tracks.begin();
         it != feature_tracks.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->matched_features_in_frames.erase(it->matched_features_in_frames.begin());
            if (it->matched_features_in_frames.size() == 0)
                feature_tracks.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature_tracks.begin(), it_next = feature_tracks.begin(); it != feature_tracks.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->matched_features_in_frames.erase(it->matched_features_in_frames.begin() + j);
            if (it->matched_features_in_frames.size() == 0)
                feature_tracks.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeatureOccurrencesAcrossFrames &feature_track, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeatureObservation &frame_i = feature_track.matched_features_in_frames[frame_count - 2 - feature_track.start_frame];
    const FeatureObservation &frame_j = feature_track.matched_features_in_frames[frame_count - 1 - feature_track.start_frame];

    double compensated_parallax = 0;

    double u_j = frame_j.point(0);
    double v_j = frame_j.point(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = rotations_camera_to_imu[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * rotations_camera_to_imu[camera_id_i] * p_i;
    p_i_comp = p_i;
    double depth_i = p_i(2);
    double u_i = p_i(0) / depth_i;
    double v_i = p_i(1) / depth_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    compensated_parallax = max(compensated_parallax, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return compensated_parallax;
}