/*
 * FAR Planner
 * Copyright (C) 2021 Fan Yang - All rights reserved
 * fanyang2@andrew.cmu.edu,
 */

#include "far_planner/map_handler.h"

/***************************************************************************************/

void MapHandler::Init(const MapHandlerParams& params) {
    map_params_ = params;
    const int row_num = std::ceil(map_params_.grid_max_length / map_params_.cell_length);
    const int col_num = row_num;
    int level_num = std::ceil(map_params_.grid_max_height / map_params_.cell_height);
    neighbor_Lnum_ = std::ceil(map_params_.sensor_range * 2.0f / map_params_.cell_length) + 1;
    neighbor_Hnum_ = 5;
    if (level_num % 2 == 0) level_num++;            // force to odd number, robot will be at center
    if (neighbor_Lnum_ % 2 == 0) neighbor_Lnum_++;  // force to odd number

    // inlitialize grid
    Eigen::Vector3i pointcloud_grid_size(row_num, col_num, level_num);
    Eigen::Vector3d pointcloud_grid_origin(0, 0, 0);
    Eigen::Vector3d pointcloud_grid_resolution(
        map_params_.cell_length, map_params_.cell_length, map_params_.cell_height);
    PointCloudPtr cloud_ptr_tmp;
    world_obs_cloud_grid_ = std::make_unique<grid_ns::Grid<PointCloudPtr>>(
        pointcloud_grid_size, cloud_ptr_tmp, pointcloud_grid_origin, pointcloud_grid_resolution, 3);

    world_free_cloud_grid_ = std::make_unique<grid_ns::Grid<PointCloudPtr>>(
        pointcloud_grid_size, cloud_ptr_tmp, pointcloud_grid_origin, pointcloud_grid_resolution, 3);

    const int n_cell = world_obs_cloud_grid_->GetCellNumber();
    for (int i = 0; i < n_cell; i++) {
        world_obs_cloud_grid_->GetCell(i) = PointCloudPtr(new PointCloud);
        world_free_cloud_grid_->GetCell(i) = PointCloudPtr(new PointCloud);
    }
    global_visited_induces_.resize(n_cell), util_remove_check_list_.resize(n_cell);
    util_obs_modified_list_.resize(n_cell), util_free_modified_list_.resize(n_cell);
    std::fill(global_visited_induces_.begin(), global_visited_induces_.end(), 0);
    std::fill(util_obs_modified_list_.begin(), util_obs_modified_list_.end(), 0);
    std::fill(util_free_modified_list_.begin(), util_free_modified_list_.end(), 0);
    std::fill(util_remove_check_list_.begin(), util_remove_check_list_.end(), 0);

    // init terrain height map
    int height_dim =
        std::ceil((map_params_.sensor_range + map_params_.cell_length) * 3.0f / FARUtil::kLeafSize);
    // int height_dim = std::ceil(map_params_.grid_max_length / FARUtil::kLeafSize);
    if (height_dim % 2 == 0) height_dim++;
    Eigen::Vector3i height_grid_size(height_dim, height_dim, 1);
    Eigen::Vector3d height_grid_origin(0, 0, 0);
    // Eigen::Vector3d height_grid_resolution(FARUtil::robot_dim, FARUtil::robot_dim,
    // FARUtil::kLeafSize);
    Eigen::Vector3d height_grid_resolution(
        FARUtil::kLeafSize, FARUtil::kLeafSize, FARUtil::kLeafSize);
    std::vector<float> temp_vec;
    terrain_height_grid_ = std::make_unique<grid_ns::Grid<std::vector<float>>>(
        height_grid_size, temp_vec, height_grid_origin, height_grid_resolution, 3);

    float temp_vef;
    terrain_avg_height_grid_ = std::make_unique<grid_ns::Grid<float>>(
        height_grid_size, temp_vef, height_grid_origin, height_grid_resolution, 3);

    const int n_terrain_cell = terrain_height_grid_->GetCellNumber();
    terrain_grid_occupy_list_.resize(n_terrain_cell),
        terrain_grid_traverse_list_.resize(n_terrain_cell);
    std::fill(terrain_grid_occupy_list_.begin(), terrain_grid_occupy_list_.end(), 0);
    std::fill(terrain_grid_traverse_list_.begin(), terrain_grid_traverse_list_.end(), 0);

    INFLATE_N = 1;
    flat_terrain_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    risk_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    obstacle_cloud_output_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    steep_slope_cloud_output_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    moderate_slope_cloud_output_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());

    risk_cloud_rgb_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    ave_high_terrain_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    kdtree_terrain_clould_ = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());
    kdtree_terrain_clould_->setSortedResults(false);

    // 风险评估
    risk_map_ready_ = false;
    occlusion_boundary_ready = false;
    initial_robot_pos_ = Point3D(0, 0, 0);  // 会在第一次更新时设置

    // [新增] 初始化五类地形点云
    obstacle_cloud_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    occlusion_cloud_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    steep_slope_cloud_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    moderate_slope_cloud_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    flat_terrain_cloud_rgb_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());

    slope_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());

    // [新增] 初始化滞后阈值状态
    is_first_classification_frame_ = true;
}

void MapHandler::ResetGripMapCloud() {
    const int n_cell = world_obs_cloud_grid_->GetCellNumber();
    for (int i = 0; i < n_cell; i++) {
        world_obs_cloud_grid_->GetCell(i)->clear();
        world_free_cloud_grid_->GetCell(i)->clear();
    }
    std::fill(global_visited_induces_.begin(), global_visited_induces_.end(), 0);
    std::fill(util_obs_modified_list_.begin(), util_obs_modified_list_.end(), 0);
    std::fill(util_free_modified_list_.begin(), util_free_modified_list_.end(), 0);
    std::fill(util_remove_check_list_.begin(), util_remove_check_list_.end(), 0);
    std::fill(terrain_grid_occupy_list_.begin(), terrain_grid_occupy_list_.end(), 0);
    std::fill(terrain_grid_traverse_list_.begin(), terrain_grid_traverse_list_.end(), 0);
}

void MapHandler::ClearObsCellThroughPosition(const Point3D& point) {
    const Eigen::Vector3i psub = world_obs_cloud_grid_->Pos2Sub(point.x, point.y, point.z);
    std::vector<Eigen::Vector3i> ray_subs;
    world_obs_cloud_grid_->RayTraceSubs(robot_cell_sub_, psub, ray_subs);
    const int H = neighbor_Hnum_ / 2;
    for (const auto& sub : ray_subs) {
        for (int k = -H; k <= H; k++) {
            Eigen::Vector3i csub = sub;
            csub.z() += k;
            const int ind = world_obs_cloud_grid_->Sub2Ind(csub);
            if (!world_obs_cloud_grid_->InRange(csub) ||
                neighbor_obs_indices_.find(ind) == neighbor_obs_indices_.end())
                continue;
            world_obs_cloud_grid_->GetCell(ind)->clear();
            if (world_free_cloud_grid_->GetCell(ind)->empty()) {
                global_visited_induces_[ind] = 0;
            }
        }
    }
}

void MapHandler::GetCloudOfPoint(const Point3D& center, const PointCloudPtr& cloudOut,
    const CloudType& type, const bool& is_large) {
    cloudOut->clear();
    const Eigen::Vector3i sub = world_obs_cloud_grid_->Pos2Sub(center.x, center.y, center.z);
    const int N = is_large ? 1 : 0;
    const int H = neighbor_Hnum_ / 2;
    for (int i = -N; i <= N; i++) {
        for (int j = -N; j <= N; j++) {
            for (int k = -H; k <= H; k++) {
                Eigen::Vector3i csub = sub;
                csub.x() += i, csub.y() += j, csub.z() += k;
                if (!world_obs_cloud_grid_->InRange(csub)) continue;
                if (type == CloudType::FREE_CLOUD) {
                    *cloudOut += *(world_free_cloud_grid_->GetCell(csub));
                } else if (type == CloudType::OBS_CLOUD) {
                    *cloudOut += *(world_obs_cloud_grid_->GetCell(csub));
                } else {
                    if (FARUtil::IsDebug) ROS_ERROR("MH: Assigned cloud type invalid.");
                    return;
                }
            }
        }
    }
}

void MapHandler::SetMapOrigin(const Point3D& ori_robot_pos) {
    Point3D map_origin;
    const Eigen::Vector3i dim = world_obs_cloud_grid_->GetSize();
    map_origin.x = ori_robot_pos.x - (map_params_.cell_length * dim.x()) / 2.0f;
    map_origin.y = ori_robot_pos.y - (map_params_.cell_length * dim.y()) / 2.0f;
    map_origin.z = ori_robot_pos.z - (map_params_.cell_height * dim.z()) / 2.0f -
                   FARUtil::vehicle_height;  // From Ground Level
    Eigen::Vector3d pointcloud_grid_origin(map_origin.x, map_origin.y, map_origin.z);
    world_obs_cloud_grid_->SetOrigin(pointcloud_grid_origin);
    MapHandlerParams map_params_;
    world_free_cloud_grid_->SetOrigin(pointcloud_grid_origin);
    is_init_ = true;
    if (FARUtil::IsDebug) ROS_INFO("MH: Global Cloud Map Grid Initialized.");
}

// 更新机器人位置，更新邻居节点的idx
// 并更新terrain_heigh grid的坐标位置，让其跟着机器人移动
void MapHandler::UpdateRobotPosition(const Point3D& odom_pos) {
    if (!is_init_) this->SetMapOrigin(odom_pos);
    robot_cell_sub_ =
        world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(odom_pos.x, odom_pos.y, odom_pos.z));
    // Get neighbor indices
    neighbor_free_indices_.clear(), neighbor_obs_indices_.clear();
    const int N = neighbor_Lnum_ / 2;
    const int H = neighbor_Hnum_ / 2;
    Eigen::Vector3i neighbor_sub;
    for (int i = -N; i <= N; i++) {
        neighbor_sub.x() = robot_cell_sub_.x() + i;
        for (int j = -N; j <= N; j++) {
            neighbor_sub.y() = robot_cell_sub_.y() + j;
            // additional terrain points -1
            neighbor_sub.z() = robot_cell_sub_.z() - H - 1;
            if (world_obs_cloud_grid_->InRange(neighbor_sub)) {
                int ind = world_obs_cloud_grid_->Sub2Ind(neighbor_sub);
                neighbor_free_indices_.insert(ind);
            }
            for (int k = -H * 2; k <= H * 2; k++) {
                neighbor_sub.z() = robot_cell_sub_.z() + k;
                if (world_obs_cloud_grid_->InRange(neighbor_sub)) {
                    int ind = world_obs_cloud_grid_->Sub2Ind(neighbor_sub);
                    neighbor_obs_indices_.insert(ind), neighbor_free_indices_.insert(ind);
                }
            }
        }
    }
    this->SetTerrainHeightGridOrigin(odom_pos);

    // [新增] 记录初始位置
    if (initial_robot_pos_.x == 0 && initial_robot_pos_.y == 0 && initial_robot_pos_.z == 0) {
        initial_robot_pos_ = odom_pos;
    }
}

void MapHandler::SetTerrainHeightGridOrigin(const Point3D& robot_pos) {
    // update terrain height grid center
    const Eigen::Vector3d res = terrain_height_grid_->GetResolution();
    const Eigen::Vector3i dim = terrain_height_grid_->GetSize();
    Eigen::Vector3d grid_origin;
    grid_origin.x() = robot_pos.x - (res.x() * dim.x()) / 2.0f;
    grid_origin.y() = robot_pos.y - (res.y() * dim.y()) / 2.0f;
    grid_origin.z() = 0.0f - (res.z() * dim.z()) / 2.0f;
    terrain_height_grid_->SetOrigin(grid_origin);
    terrain_avg_height_grid_->SetOrigin(grid_origin);
}

void MapHandler::GetSurroundObsCloud(const PointCloudPtr& obsCloudOut) {
    if (!is_init_) return;
    obsCloudOut->clear();
    for (const auto& neighbor_ind : neighbor_obs_indices_) {
        if (world_obs_cloud_grid_->GetCell(neighbor_ind)->empty()) continue;
        *obsCloudOut += *(world_obs_cloud_grid_->GetCell(neighbor_ind));
    }
}

void MapHandler::GetSurroundFreeCloud(const PointCloudPtr& freeCloudOut) {
    if (!is_init_) return;
    freeCloudOut->clear();
    for (const auto& neighbor_ind : neighbor_free_indices_) {
        if (world_free_cloud_grid_->GetCell(neighbor_ind)->empty()) continue;
        *freeCloudOut += *(world_free_cloud_grid_->GetCell(neighbor_ind));
    }
}

void MapHandler::UpdateObsCloudGrid(const PointCloudPtr& obsCloudInOut) {
    if (!is_init_ || obsCloudInOut->empty()) return;
    std::fill(util_obs_modified_list_.begin(), util_obs_modified_list_.end(), 0);
    PointCloudPtr obs_valid_ptr(new pcl::PointCloud<PCLPoint>());
    for (const auto& point : obsCloudInOut->points) {
        Eigen::Vector3i sub =
            world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
        if (!world_obs_cloud_grid_->InRange(sub)) continue;
        const int ind = world_obs_cloud_grid_->Sub2Ind(sub);
        if (neighbor_obs_indices_.find(ind) != neighbor_obs_indices_.end()) {
            world_obs_cloud_grid_->GetCell(ind)->points.push_back(point);
            obs_valid_ptr->points.push_back(point);
            util_obs_modified_list_[ind] = 1;
            global_visited_induces_[ind] = 1;
        }
    }
    *obsCloudInOut = *obs_valid_ptr;
    // Filter Modified Ceils
    for (int i = 0; i < world_obs_cloud_grid_->GetCellNumber(); ++i) {
        if (util_obs_modified_list_[i] == 1)
            FARUtil::FilterCloud(world_obs_cloud_grid_->GetCell(i), FARUtil::kLeafSize);
    }
}

void MapHandler::UpdateFreeCloudGrid(const PointCloudPtr& freeCloudIn) {
    if (!is_init_ || freeCloudIn->empty()) return;
    std::fill(util_free_modified_list_.begin(), util_free_modified_list_.end(), 0);
    for (const auto& point : freeCloudIn->points) {
        Eigen::Vector3i sub =
            world_free_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
        if (!world_free_cloud_grid_->InRange(sub)) continue;
        const int ind = world_free_cloud_grid_->Sub2Ind(sub);
        world_free_cloud_grid_->GetCell(ind)->points.push_back(point);
        util_free_modified_list_[ind] = 1;
        global_visited_induces_[ind] = 1;
    }
    // Filter Modified Ceils
    for (int i = 0; i < world_free_cloud_grid_->GetCellNumber(); ++i) {
        if (util_free_modified_list_[i] == 1)
            FARUtil::FilterCloud(world_free_cloud_grid_->GetCell(i), FARUtil::kLeafSize);
    }
}
// 获得对应点的高度信息
float MapHandler::TerrainHeightOfPoint(const Point3D& p, bool& is_matched, const bool& is_search) {
    is_matched = false;
    const Eigen::Vector3i sub = terrain_height_grid_->Pos2Sub(Eigen::Vector3d(p.x, p.y, 0.0f));
    if (terrain_height_grid_->InRange(sub)) {
        const int ind = terrain_height_grid_->Sub2Ind(sub);
        if (terrain_grid_traverse_list_[ind] != 0) {
            is_matched = true;
            return terrain_avg_height_grid_->GetCell(ind);
        }
    }
    if (is_search) {
        float matched_dist_squre;
        const float terrain_h = NearestHeightOfPoint(p, matched_dist_squre);
        return terrain_h;
    }
    return p.z;
}

float MapHandler::NearestTerrainHeightofNavPoint(const Point3D& point, bool& is_associated) {
    const float p_th = point.z - FARUtil::vehicle_height;
    const Eigen::Vector3i ori_sub =
        world_free_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, p_th));
    is_associated = false;
    if (world_free_cloud_grid_->InRange(ori_sub)) {
        // downward seach
        bool is_dw_associated = false;
        Eigen::Vector3i dw_near_sub = ori_sub;
        float dw_terrain_h = p_th;
        while (world_free_cloud_grid_->InRange(dw_near_sub)) {
            if (!world_free_cloud_grid_->GetCell(dw_near_sub)->empty()) {
                int counter = 0;
                dw_terrain_h = 0.0f;
                for (const auto& pcl_p : world_free_cloud_grid_->GetCell(dw_near_sub)->points) {
                    dw_terrain_h += pcl_p.z, counter++;
                }
                dw_terrain_h /= (float)counter;
                is_dw_associated = true;
                break;
            }
            dw_near_sub.z()--;
        }
        // upward search
        bool is_up_associated = false;
        Eigen::Vector3i up_near_sub = ori_sub;
        float up_terrain_h = p_th;
        while (world_free_cloud_grid_->InRange(up_near_sub)) {
            if (!world_free_cloud_grid_->GetCell(up_near_sub)->empty()) {
                int counter = 0;
                up_terrain_h = 0.0f;
                for (const auto& pcl_p : world_free_cloud_grid_->GetCell(up_near_sub)->points) {
                    up_terrain_h += pcl_p.z, counter++;
                }
                up_terrain_h /= (float)counter;
                is_up_associated = true;
                break;
            }
            up_near_sub.z()++;
        }
        is_associated = (is_up_associated || is_dw_associated) ? true : false;
        if (is_up_associated && is_dw_associated) {  // compare nearest
            if (up_near_sub.z() - ori_sub.z() < ori_sub.z() - dw_near_sub.z())
                return up_terrain_h;
            else
                return dw_terrain_h;
        } else if (is_up_associated)
            return up_terrain_h;
        else
            return dw_terrain_h;
    }
    return p_th;
}
// 判断导航点是不是在地形相关的领域网络中
// 就是是否在neighbor_obs_indices_索引范围内或者extend_obs_indices_
bool MapHandler::IsNavPointOnTerrainNeighbor(const Point3D& point, const bool& is_extend) {
    const float h = 0;
    const Eigen::Vector3i sub =
        world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, h));
    if (!world_obs_cloud_grid_->InRange(sub)) return false;
    const int ind = world_obs_cloud_grid_->Sub2Ind(sub);
    if (is_extend && extend_obs_indices_.find(ind) != extend_obs_indices_.end()) {
        return true;
    }
    if (!is_extend && neighbor_obs_indices_.find(ind) != neighbor_obs_indices_.end()) {
        return true;
    }
    return false;
}
// 调整导航点的高度
void MapHandler::AdjustNodesHeight(const NodePtrStack& nodes) {
    if (nodes.empty()) return;
    for (const auto& node_ptr : nodes) {
        if (!node_ptr->is_active || node_ptr->is_boundary || FARUtil::IsFreeNavNode(node_ptr) ||
            FARUtil::IsOutsideGoal(node_ptr) ||
            !FARUtil::IsPointInLocalRange(node_ptr->position, true)) {
            continue;
        }
        bool is_match = false;
        float terrain_h = TerrainHeightOfPoint(node_ptr->position, is_match, false);
        if (is_match) {
            terrain_h += FARUtil::vehicle_height;
            if (node_ptr->pos_filter_vec.empty()) {
                node_ptr->position.z = terrain_h;
            } else {
                node_ptr->pos_filter_vec.back().z = terrain_h;  // assign to position filter
                node_ptr->position.z = FARUtil::AveragePoints(node_ptr->pos_filter_vec).z;
            }
        }
    }
}

void MapHandler::AdjustCTNodeHeight(const CTNodeStack& ctnodes) {
    if (ctnodes.empty()) return;
    const float H_MAX = FARUtil::robot_pos.z + FARUtil::kTolerZ;
    const float H_MIN = FARUtil::robot_pos.z - FARUtil::kTolerZ;
    for (auto& ctnode_ptr : ctnodes) {
        float min_th, max_th;
        const float avg_h = NearestHeightOfRadius(ctnode_ptr->position, FARUtil::kMatchDist, min_th,
            max_th, ctnode_ptr->is_ground_associate);
        if (ctnode_ptr->is_ground_associate) {
            // ctnode_ptr->position.z = min_th + FARUtil::vehicle_height;
            // ctnode_ptr->position.z = std::max(std::min(ctnode_ptr->position.z, H_MAX), H_MIN);
            ctnode_ptr->position.z += FARUtil::vehicle_height;
        } else {
            ctnode_ptr->position.z =
                TerrainHeightOfPoint(ctnode_ptr->position, ctnode_ptr->is_ground_associate, true);
            ctnode_ptr->position.z += FARUtil::vehicle_height;
            // ctnode_ptr->position.z = std::max(std::min(ctnode_ptr->position.z, H_MAX), H_MIN);
        }
    }
}

void MapHandler::ObsNeighborCloudWithTerrain(
    std::unordered_set<int>& neighbor_obs, std::unordered_set<int>& extend_terrain_obs) {
    std::unordered_set<int> neighbor_copy = neighbor_obs;
    neighbor_obs.clear();
    const float R = map_params_.cell_length * 0.7071f;  // sqrt(2)/2
    for (const auto& idx : neighbor_copy) {
        const Point3D pos = Point3D(world_obs_cloud_grid_->Ind2Pos(idx));
        const Eigen::Vector3i sub =
            terrain_height_grid_->Pos2Sub(Eigen::Vector3d(pos.x, pos.y, 0.0f));
        const int terrain_ind = terrain_height_grid_->Sub2Ind(sub);
        bool inRange = false;
        float minH, maxH;
        const float avgH = NearestHeightOfRadius(pos, R, minH, maxH, inRange);
        if (inRange && pos.z + map_params_.cell_height > minH &&
            pos.z - map_params_.cell_height <
                maxH + FARUtil::kTolerZ)  // use map_params_.cell_height/2.0 as a tolerance margin
        {
            neighbor_obs.insert(idx);
        }
    }
    extend_terrain_obs.clear();  // assign extended terrain obs indices
    const std::vector<int> inflate_vec{-1, 0};
    for (const int& idx : neighbor_obs) {
        const Eigen::Vector3i csub = world_obs_cloud_grid_->Ind2Sub(idx);
        for (const int& plus : inflate_vec) {
            Eigen::Vector3i sub = csub;
            sub.z() += plus;
            if (!world_obs_cloud_grid_->InRange(sub)) continue;
            const int plus_idx = world_obs_cloud_grid_->Sub2Ind(sub);
            extend_terrain_obs.insert(plus_idx);
        }
    }
}

void MapHandler::UpdateTerrainHeightGrid(
    const PointCloudPtr& freeCloudIn, const PointCloudPtr& terrainHeightOut) {
    if (freeCloudIn->empty()) return;
    PointCloudPtr copy_free_ptr(new pcl::PointCloud<PCLPoint>());
    pcl::copyPointCloud(*freeCloudIn, *copy_free_ptr);
    // FARUtil::FilterCloud(copy_free_ptr, terrain_height_grid_->GetResolution());
    std::fill(terrain_grid_occupy_list_.begin(), terrain_grid_occupy_list_.end(), 0);
    for (const auto& point : copy_free_ptr->points) {
        Eigen::Vector3i csub =
            terrain_height_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, 0.0f));
        std::vector<Eigen::Vector3i> subs;
        this->Expansion2D(csub, subs, INFLATE_N);
        for (const auto& sub : subs) {
            if (!terrain_height_grid_->InRange(sub)) continue;
            const int ind = terrain_height_grid_->Sub2Ind(sub);
            if (terrain_grid_occupy_list_[ind] == 0) {
                terrain_height_grid_->GetCell(ind).resize(1);
                terrain_height_grid_->GetCell(ind)[0] = point.z;
            } else {
                terrain_height_grid_->GetCell(ind).push_back(point.z);
            }
            terrain_grid_occupy_list_[ind] = 1;
        }
    }
    const int N = terrain_grid_occupy_list_.size();
    CalculateAveHigh();
    ComputeTerrainRiskAttributes(terrainHeightOut);
    // 判断terrain_grid_traverse_list_
    terrainHeightOut->header.frame_id = FARUtil::worldFrameId;  // 设置为世界坐标系
    terrainHeightOut->header.stamp = pcl_conversions::toPCL(ros::Time::now());
    if (terrainHeightOut->empty()) {  // set terrain height kdtree
        FARUtil::ClearKdTree(flat_terrain_cloud_, kdtree_terrain_clould_);
    } else {
        // this->AssignFlatTerrainCloud(terrainHeightOut, flat_terrain_cloud_);
        // kdtree_terrain_clould_->setInputCloud(flat_terrain_cloud_);
        kdtree_terrain_clould_->setInputCloud(ave_high_terrain_cloud_);
    }
    // update surrounding obs cloud grid indices based on terrain
    this->ObsNeighborCloudWithTerrain(neighbor_obs_indices_, extend_obs_indices_);
}

void MapHandler::CalculateAveHigh() {
    const Eigen::Vector3i robot_sub = terrain_height_grid_->Pos2Sub(
        Eigen::Vector3d(FARUtil::robot_pos.x, FARUtil::robot_pos.y, 0.0f));
    ave_high_terrain_cloud_->clear();
    if (!terrain_height_grid_->InRange(robot_sub)) {
        ROS_ERROR("MH: terrain height analysis error: robot position is not in range");
        return;
    }
    const int n_cells = terrain_height_grid_->GetCellNumber();
    for (int i = 0; i < n_cells; i++) {
        // 跳过没有高度数据的网格
        if (terrain_grid_occupy_list_[i] == 0) continue;

        const auto& height_vec = terrain_height_grid_->GetCell(i);
        if (height_vec.size() < 1) continue;

        // 使用中位数替代平均值，抗离群点噪声
        float avg_height = 0.0f;
        if (height_vec.size() == 1) {
            avg_height = height_vec[0];
        } else {
            // 复制数据用于排序（不修改原始 grid）
            std::vector<float> heights_copy(height_vec.begin(), height_vec.end());
            std::nth_element(heights_copy.begin(), heights_copy.begin() + heights_copy.size() / 2,
                heights_copy.end());
            avg_height = heights_copy[heights_copy.size() / 2];
        }

        // 获取网格的3D位置
        Eigen::Vector3d grid_pos = terrain_height_grid_->Ind2Pos(i);

        // 创建点云点，使用平均高度
        PCLPoint point;
        point.x = grid_pos.x();
        point.y = grid_pos.y();
        point.z = avg_height;          // 使用计算出的平均高度
        point.intensity = avg_height;  // 可选：将高度信息也存储在intensity字段

        // 添加到输出点云
        ave_high_terrain_cloud_->points.push_back(point);
        terrain_avg_height_grid_->GetCell(i) = avg_height;
    }
    ave_high_terrain_cloud_->width = ave_high_terrain_cloud_->points.size();
    ave_high_terrain_cloud_->height = 1;
    ave_high_terrain_cloud_->is_dense = false;
    ave_high_terrain_cloud_->header.frame_id = FARUtil::worldFrameId;
    ave_high_terrain_cloud_->header.stamp = pcl_conversions::toPCL(ros::Time::now());
}

void MapHandler::GetNeighborCeilsCenters(PointStack& neighbor_centers) {
    if (!is_init_) return;
    neighbor_centers.clear();
    for (const auto& ind : neighbor_obs_indices_) {
        if (global_visited_induces_[ind] == 0) continue;
        Point3D center_p(world_obs_cloud_grid_->Ind2Pos(ind));
        neighbor_centers.push_back(center_p);
    }
}

void MapHandler::GetOccupancyCeilsCenters(PointStack& occupancy_centers) {
    if (!is_init_) return;
    occupancy_centers.clear();
    const int N = world_obs_cloud_grid_->GetCellNumber();
    for (int ind = 0; ind < N; ind++) {
        if (global_visited_induces_[ind] == 0) continue;
        Point3D center_p(world_obs_cloud_grid_->Ind2Pos(ind));
        occupancy_centers.push_back(center_p);
    }
}

void MapHandler::RemoveObsCloudFromGrid(const PointCloudPtr& obsCloud) {
    std::fill(util_remove_check_list_.begin(), util_remove_check_list_.end(), 0);
    for (const auto& point : obsCloud->points) {
        Eigen::Vector3i sub =
            world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
        if (!world_free_cloud_grid_->InRange(sub)) continue;
        const int ind = world_free_cloud_grid_->Sub2Ind(sub);
        util_remove_check_list_[ind] = 1;
    }
    for (const auto& ind : neighbor_obs_indices_) {
        if (util_remove_check_list_[ind] == 1 && global_visited_induces_[ind] == 1) {
            FARUtil::RemoveOverlapCloud(world_obs_cloud_grid_->GetCell(ind), obsCloud);
        }
    }
}

void MapHandler::GridToImg(cv::Mat& height_img, cv::Mat& var_img, cv::Mat& mask_img) {
    const Eigen::Vector3i dim = terrain_height_grid_->GetSize();
    height_img =
        cv::Mat(dim.y(), dim.x(), CV_32FC1, cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    var_img = cv::Mat::zeros(dim.y(), dim.x(), CV_32FC1);
    mask_img = cv::Mat::zeros(dim.y(), dim.x(), CV_8UC1);

    // [新增] 记录点云密度
    point_density_mat_ = cv::Mat::zeros(dim.y(), dim.x(), CV_32FC1);

    for (const auto& point : ave_high_terrain_cloud_->points) {
        Eigen::Vector3i sub =
            terrain_height_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, 0.0f));
        if (!terrain_height_grid_->InRange(sub)) continue;

        int r = sub.y();
        int c = sub.x();

        height_img.at<float>(r, c) = point.z;

        // 格内落差需要从原始 grid 获取
        int ind = terrain_height_grid_->Sub2Ind(sub);
        const auto& heights = terrain_height_grid_->GetCell(ind);
        if (!heights.empty()) {
            float min_z = *std::min_element(heights.begin(), heights.end());
            float max_z = *std::max_element(heights.begin(), heights.end());
            var_img.at<float>(r, c) = max_z - min_z;

            // [新增] 记录点云数量
            point_density_mat_.at<float>(r, c) = heights.size();
        } else {
            var_img.at<float>(r, c) = std::numeric_limits<float>::quiet_NaN();
        }

        mask_img.at<uchar>(r, c) = 255;
    }
}

// 计算可通行图
void MapHandler::ComputeTerrainRiskAttributes(const PointCloudPtr& terrainHeightOut) {
    if (!terrain_height_grid_) return;

    // ============================================================
    // 步骤 1: 生成基础数据
    // ============================================================
    GridToImg(raw_h, inner_diff, valid_mask);

    // 填补空洞（形态学闭运算）  先膨胀再腐蚀
    cv::Mat closed_mask;
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(valid_mask, closed_mask, cv::MORPH_CLOSE, close_kernel);

    const float SLOPE_THRESHOLD = 0.5f;  // 坡度阈值
    const float STEP_THRESHOLD = 0.3f;   // 高度差阈值

    // ============================================================
    // 步骤 2: 检测遮挡边界
    // ============================================================

    int center_r = raw_h.rows / 2;
    int center_c = raw_h.cols / 2;
    float res = terrain_height_grid_->GetResolution().x();

    occlusion_boundary_mask_ = cv::Mat::zeros(closed_mask.rows, closed_mask.cols, CV_8UC1);
    const float MIN_DENSITY = 1.0f;  // 降低密度阈值

    // [方法1] 从有效格子出发，检测边界
    for (int r = 1; r < closed_mask.rows - 1; r++) {
        for (int c = 1; c < closed_mask.cols - 1; c++) {
            if (closed_mask.at<uchar>(r, c) == 255 &&
                point_density_mat_.at<float>(r, c) >= MIN_DENSITY) {
                // 检查8邻域（不只是4邻域）
                bool has_invalid_neighbor = false;
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) continue;
                        if (closed_mask.at<uchar>(r + dr, c + dc) == 0) {
                            has_invalid_neighbor = true;
                            break;
                        }
                    }
                    if (has_invalid_neighbor) break;
                }

                if (has_invalid_neighbor) {
                    occlusion_boundary_mask_.at<uchar>(r, c) = 255;
                }
            }
        }
    }

    // [方法2] 从无效格子出发，向外标记一圈边界（关键！）
    for (int r = 1; r < closed_mask.rows - 1; r++) {
        for (int c = 1; c < closed_mask.cols - 1; c++) {
            // 如果当前是无效格子
            if (closed_mask.at<uchar>(r, c) == 0) {
                // 检查8邻域是否有有效格子
                for (int dr = -1; dr <= 1; dr++) {
                    for (int dc = -1; dc <= 1; dc++) {
                        if (dr == 0 && dc == 0) continue;
                        int nr = r + dr;
                        int nc = c + dc;

                        // 如果邻居是有效格子，标记为边界
                        if (closed_mask.at<uchar>(nr, nc) == 255 &&
                            point_density_mat_.at<float>(nr, nc) >= MIN_DENSITY) {
                            occlusion_boundary_mask_.at<uchar>(nr, nc) = 255;
                        }
                    }
                }
            }
        }
    }

    // 膨胀遮挡边界
    cv::Mat dilate_occlusion_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(occlusion_boundary_mask_, occlusion_boundary_mask_, dilate_occlusion_kernel);

    if (FARUtil::IsDebug) {
        ROS_INFO_THROTTLE(
            2.0, "MH: Occlusion boundary cells: %d", cv::countNonZero(occlusion_boundary_mask_));
    }

    if (!occlusion_boundary_ready) {
        float dist_moved = std::sqrt(std::pow(FARUtil::robot_pos.x - initial_robot_pos_.x, 2) +
                                     std::pow(FARUtil::robot_pos.y - initial_robot_pos_.y, 2));

        if (dist_moved < 1.5f) {
            // 获取机器人在地形网格中的位置
            Eigen::Vector3i robot_sub = terrain_height_grid_->Pos2Sub(
                Eigen::Vector3d(FARUtil::robot_pos.x, FARUtil::robot_pos.y, 0.0f));
            int robot_r = robot_sub.y();
            int robot_c = robot_sub.x();

            const float CLEAR_RADIUS = 1.8f;
            for (int r = 0; r < occlusion_boundary_mask_.rows; r++) {
                for (int c = 0; c < occlusion_boundary_mask_.cols; c++) {
                    // 计算格子到机器人的实际距离
                    float dist =
                        std::sqrt(std::pow(r - robot_r, 2) + std::pow(c - robot_c, 2)) * res;
                    if (dist < CLEAR_RADIUS) {
                        occlusion_boundary_mask_.at<uchar>(r, c) = 0;
                    }
                }
            }
            if (FARUtil::IsDebug) {
                ROS_INFO_THROTTLE(1.0,
                    "MH: Occlusion boundary suppressed around robot (radius %.1fm, moved %.2fm)",
                    CLEAR_RADIUS, dist_moved);
            }
        } else {
            occlusion_boundary_ready = true;
            if (FARUtil::IsDebug) {
                ROS_INFO(
                    "MH: Occlusion boundary detection activated after %.2fm movement", dist_moved);
            }
        }
    }

    // ============================================================
    // 步骤 3: 填补高度图并计算坡度
    // ============================================================
    cv::Mat filled_h = raw_h.clone();

    // 改进填补策略：只填补被完全包围的小洞
    for (int r = 2; r < raw_h.rows - 2; r++) {
        for (int c = 2; c < raw_h.cols - 2; c++) {
            float center_val = filled_h.at<float>(r, c);

            // 跳过已有有效值的格子
            if (!std::isnan(center_val)) continue;

            // 检查5x5邻域，只有被有效数据"包围"的洞才填补
            int valid_neighbor_count = 0;
            float sum = 0.0f;

            for (int nr = -2; nr <= 2; nr++) {
                for (int nc = -2; nc <= 2; nc++) {
                    if (nr == 0 && nc == 0) continue;
                    float neighbor_val = raw_h.at<float>(r + nr, c + nc);
                    if (!std::isnan(neighbor_val)) {
                        sum += neighbor_val;
                        valid_neighbor_count++;
                    }
                }
            }

            // 只有周围至少有20个(5x5=25个-1)有效点，才认为是"被包围的洞"
            if (valid_neighbor_count >= 20) {
                filled_h.at<float>(r, c) = sum / valid_neighbor_count;
            }
            // 否则保持 NaN，让后续 Sobel 自动忽略
        }
    }

    // [新增] 在计算 Sobel 之前，把 NaN 替换为最近有效值（仅用于 Sobel 计算）
    cv::Mat sobel_input = filled_h.clone();
    for (int r = 0; r < sobel_input.rows; r++) {
        for (int c = 0; c < sobel_input.cols; c++) {
            if (std::isnan(sobel_input.at<float>(r, c))) {
                // 向左搜索最近的有效值
                float nearest_val = 0.0f;
                for (int search_c = c - 1; search_c >= 0; search_c--) {
                    if (!std::isnan(filled_h.at<float>(r, search_c))) {
                        nearest_val = filled_h.at<float>(r, search_c);
                        break;
                    }
                }
                sobel_input.at<float>(r, c) = nearest_val;
            }
        }
    }

    // 计算坡度（使用处理后的输入）

    cv::Sobel(sobel_input, grad_x, CV_32F, 1, 0, 3, 1.0 / (8.0 * res));
    cv::Sobel(sobel_input, grad_y, CV_32F, 0, 1, 3, 1.0 / (8.0 * res));
    cv::magnitude(grad_x, grad_y, slope_mat);

    // [新增] 基于邻域完整性过滤坡度
    // [新增] 邻域完整性掩膜（0-255表示可信度）
    cv::Mat slope_confidence_mask = cv::Mat::ones(slope_mat.rows, slope_mat.cols, CV_32FC1);

    for (int r = 2; r < slope_mat.rows - 2; r++) {
        for (int c = 2; c < slope_mat.cols - 2; c++) {
            if (closed_mask.at<uchar>(r, c) == 0) {
                slope_mat.at<float>(r, c) = 0.0f;
                slope_confidence_mask.at<float>(r, c) = 0.0f;
                continue;
            }

            // 检查3x3邻域的完整性
            int valid_neighbor_count = 0;
            for (int nr = -1; nr <= 1; nr++) {
                for (int nc = -1; nc <= 1; nc++) {
                    if (closed_mask.at<uchar>(r + nr, c + nc) == 255) {
                        valid_neighbor_count++;
                    }
                }
            }

            float confidence = valid_neighbor_count / 9.0f;

            // 额外检查：如果5x5邻域稀疏，进一步降低
            int extended_valid_count = 0;
            for (int nr = -2; nr <= 2; nr++) {
                for (int nc = -2; nc <= 2; nc++) {
                    if (r + nr >= 0 && r + nr < closed_mask.rows && c + nc >= 0 &&
                        c + nc < closed_mask.cols && closed_mask.at<uchar>(r + nr, c + nc) == 255) {
                        extended_valid_count++;
                    }
                }
            }

            if (extended_valid_count < 20) {  // 5x5有25个格子，至少要有20个
                float extended_confidence = extended_valid_count / 25.0f;
                confidence *= extended_confidence;
            }

            // 保存可信度，但不直接修改 slope_mat
            slope_confidence_mask.at<float>(r, c) = confidence;

            // [关键改进] 只对低可信度 + 高坡度的情况进行衰减
            // 避免把真实陡坡也衰减掉
            if (confidence < 0.7f && slope_mat.at<float>(r, c) > SLOPE_THRESHOLD) {
                slope_mat.at<float>(r, c) *= confidence;
            }
        }
    }

    if (FARUtil::IsDebug) {
        ROS_INFO_THROTTLE(2.0, "MH: Neighborhood integrity filtering applied");
    }

    // ============================================================
    // 步骤 4: 生成初步风险掩膜
    // ============================================================

    cv::threshold(slope_mat, slope_risk, SLOPE_THRESHOLD, 255, cv::THRESH_BINARY);
    cv::threshold(inner_diff, step_risk, STEP_THRESHOLD, 255, cv::THRESH_BINARY);

    slope_risk.convertTo(slope_risk, CV_8UC1);
    step_risk.convertTo(step_risk, CV_8UC1);

    // 去除红色噪点（开运算）
    cv::Mat remove_noise_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(step_risk, step_risk, cv::MORPH_OPEN, remove_noise_kernel);

    // ============================================================
    // 步骤 5: 初始化阶段屏蔽
    // ============================================================
    if (!risk_map_ready_) {
        float dist_moved = std::sqrt(std::pow(FARUtil::robot_pos.x - initial_robot_pos_.x, 2) +
                                     std::pow(FARUtil::robot_pos.y - initial_robot_pos_.y, 2));

        if (dist_moved < 1.0f) {
            const float CLEAR_RADIUS = 1.5f;
            for (int r = 0; r < slope_risk.rows; r++) {
                for (int c = 0; c < slope_risk.cols; c++) {
                    float dist =
                        std::sqrt(std::pow(r - center_r, 2) + std::pow(c - center_c, 2)) * res;
                    if (dist < CLEAR_RADIUS) {
                        slope_risk.at<uchar>(r, c) = 0;
                    }
                }
            }
            if (FARUtil::IsDebug) {
                ROS_INFO_THROTTLE(1.0, "MH: Slope risk suppressed (radius %.1fm, moved %.2fm)",
                    CLEAR_RADIUS, dist_moved);
            }
        } else {
            risk_map_ready_ = true;
            if (FARUtil::IsDebug) {
                ROS_INFO("MH: Risk map activated after %.2fm movement", dist_moved);
            }
        }
    }

    // ============================================================
    // 步骤 6: 红色吞噬黄色（消除墙壁周围虚假陡坡）
    // ============================================================
    cv::Mat dilate_step_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat dilated_step_risk;
    cv::dilate(step_risk, dilated_step_risk, dilate_step_kernel);

    cv::Mat not_red_zone;
    cv::bitwise_not(dilated_step_risk, not_red_zone);
    cv::bitwise_and(slope_risk, not_red_zone, slope_risk);

    // 合并最终风险
    cv::bitwise_or(slope_risk, step_risk, final_risk);

    // 过滤地图边缘噪声
    cv::Mat safe_zone;
    cv::Mat erode_boundary_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::erode(valid_mask, safe_zone, erode_boundary_kernel);
    cv::bitwise_and(final_risk, safe_zone, risk_mask_mat_);

    // ============================================================
    // 步骤 7: 生成五类地形掩膜（带滞后阈值）
    // ============================================================
    obstacle_mask_ = cv::Mat::zeros(valid_mask.rows, valid_mask.cols, CV_8UC1);
    steep_slope_mask_ = cv::Mat::zeros(valid_mask.rows, valid_mask.cols, CV_8UC1);
    moderate_slope_mask_ = cv::Mat::zeros(valid_mask.rows, valid_mask.cols, CV_8UC1);
    flat_terrain_mask_ = cv::Mat::zeros(valid_mask.rows, valid_mask.cols, CV_8UC1);

    const float SLOPE_LOW = 0.42f;   // 滞后下限
    const float SLOPE_HIGH = 0.48f;  // 滞后上限
    const float MODERATE_MIN = 0.2f;

    for (int r = 0; r < valid_mask.rows; r++) {
        for (int c = 0; c < valid_mask.cols; c++) {
            float slope_val = slope_mat.at<float>(r, c);
            float step_val = inner_diff.at<float>(r, c);

            // 优先级1: 障碍物
            if (step_risk.at<uchar>(r, c) > 100) {
                obstacle_mask_.at<uchar>(r, c) = 255;
                occlusion_boundary_mask_.at<uchar>(r, c) = 0;  // 障碍物覆盖遮挡
                continue;
            }

            // 优先级2: 陡坡（带滞后）
            bool is_steep = slope_risk.at<uchar>(r, c) > 100;
            if (!is_first_classification_frame_ && prev_steep_slope_mask_.at<uchar>(r, c) > 100) {
                // 上一帧是陡坡，需要降到下限才变缓坡
                is_steep = (slope_val > SLOPE_LOW);
            } else if (!is_first_classification_frame_ &&
                       prev_moderate_slope_mask_.at<uchar>(r, c) > 100) {
                // 上一帧是缓坡，需要升到上限才变陡坡
                is_steep = (slope_val > SLOPE_HIGH);
            }

            if (is_steep && (dilated_step_risk.at<uchar>(r, c) < 100)) {
                steep_slope_mask_.at<uchar>(r, c) = 255;
                continue;
            }

            // 优先级3: 缓坡
            if (slope_val >= MODERATE_MIN && slope_val < SLOPE_HIGH && step_val < STEP_THRESHOLD) {
                moderate_slope_mask_.at<uchar>(r, c) = 255;
                continue;
            }

            // 优先级4: 平地
            if (valid_mask.at<uchar>(r, c) == 255) {
                flat_terrain_mask_.at<uchar>(r, c) = 255;
            }
        }
    }

    // 轻度形态学处理（融合碎片，但不过度）
    cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));  // 缩小到3x3
    cv::morphologyEx(steep_slope_mask_, steep_slope_mask_, cv::MORPH_CLOSE, morph_kernel);
    cv::morphologyEx(moderate_slope_mask_, moderate_slope_mask_, cv::MORPH_CLOSE, morph_kernel);

    // [新增] 解决重叠问题:按优先级清除冲突
    for (int r = 0; r < valid_mask.rows; r++) {
        for (int c = 0; c < valid_mask.cols; c++) {
            // 障碍物优先级最高
            if (obstacle_mask_.at<uchar>(r, c) > 100) {
                occlusion_boundary_mask_.at<uchar>(r, c) = 0;
                steep_slope_mask_.at<uchar>(r, c) = 0;
                moderate_slope_mask_.at<uchar>(r, c) = 0;
                flat_terrain_mask_.at<uchar>(r, c) = 0;
            } else if (occlusion_boundary_mask_.at<uchar>(r, c) > 100) {
                steep_slope_mask_.at<uchar>(r, c) = 0;
                moderate_slope_mask_.at<uchar>(r, c) = 0;
                flat_terrain_mask_.at<uchar>(r, c) = 0;
            }
            // 陡坡优先级第二
            else if (steep_slope_mask_.at<uchar>(r, c) > 100) {
                moderate_slope_mask_.at<uchar>(r, c) = 0;
                flat_terrain_mask_.at<uchar>(r, c) = 0;
            }
            // 缓坡优先级第三
            else if (moderate_slope_mask_.at<uchar>(r, c) > 100) {
                flat_terrain_mask_.at<uchar>(r, c) = 0;
            }
        }
    }

    // 保存当前帧状态
    prev_steep_slope_mask_ = steep_slope_mask_.clone();
    prev_moderate_slope_mask_ = moderate_slope_mask_.clone();
    is_first_classification_frame_ = false;

    // ============================================================
    // 步骤 8: 生成五类地形点云
    // ============================================================
    obstacle_cloud_->clear();
    occlusion_cloud_->clear();
    steep_slope_cloud_->clear();
    moderate_slope_cloud_->clear();
    flat_terrain_cloud_rgb_->clear();

    obstacle_cloud_output_->clear();
    steep_slope_cloud_output_->clear();
    moderate_slope_cloud_output_->clear();

    // [新增] 获取当前时间戳
    const float current_time = ros::Time::now().toSec() - FARUtil::systemStartTime;

    for (int r = 0; r < valid_mask.rows; r++) {
        for (int c = 0; c < valid_mask.cols; c++) {
            int ind = terrain_height_grid_->Sub2Ind(c, r, 0);
            Eigen::Vector3d pos = terrain_height_grid_->Ind2Pos(ind);

            PointRGB p;
            p.x = pos.x();
            p.y = pos.y();
            p.z = raw_h.at<float>(r, c);

            PCLPoint pc;
            pc.x = pos.x();
            pc.y = pos.y();
            pc.z = raw_h.at<float>(r, c);
            pc.intensity = current_time;  // [新增] 时间戳

            if (obstacle_mask_.at<uchar>(r, c) > 100) {
                p.r = 255;
                p.g = 0;
                p.b = 0;
                obstacle_cloud_->push_back(p);

                obstacle_cloud_output_->push_back(pc);
            } else if (occlusion_boundary_mask_.at<uchar>(r, c) > 100) {
                bool is_map_edge =
                    (r < 5 || r >= valid_mask.rows - 5 || c < 5 || c >= valid_mask.cols - 5);
                if (is_map_edge) {
                    p.r = 128;
                    p.g = 128;
                    p.b = 128;
                } else {
                    p.r = 0;
                    p.g = 255;
                    p.b = 0;
                }
                occlusion_cloud_->push_back(p);
            } else if (steep_slope_mask_.at<uchar>(r, c) > 100) {
                p.r = 255;
                p.g = 255;
                p.b = 0;
                steep_slope_cloud_->push_back(p);

                steep_slope_cloud_output_->push_back(pc);
            } else if (moderate_slope_mask_.at<uchar>(r, c) > 100) {
                p.r = 0;
                p.g = 0;
                p.b = 255;
                moderate_slope_cloud_->push_back(p);
                moderate_slope_cloud_output_->push_back(pc);
            } else if (flat_terrain_mask_.at<uchar>(r, c) > 100) {
                p.r = 255;
                p.g = 255;
                p.b = 255;
                flat_terrain_cloud_rgb_->push_back(p);
            }
        }
    }
    // [新增] 过滤 NaN 点
    if (!obstacle_cloud_output_->empty()) {
        FARUtil::RemoveNanInfPoints(obstacle_cloud_output_);
    }
    if (!steep_slope_cloud_output_->empty()) {
        FARUtil::RemoveNanInfPoints(steep_slope_cloud_output_);
    }
    if (!moderate_slope_cloud_output_->empty()) {
        FARUtil::RemoveNanInfPoints(moderate_slope_cloud_output_);
    }

    if (FARUtil::IsDebug) {
        ROS_INFO_THROTTLE(2.0,
            "MH: Terrain - Obstacle:%lu Occlusion:%lu Steep:%lu Moderate:%lu Flat:%lu",
            obstacle_cloud_->size(), occlusion_cloud_->size(), steep_slope_cloud_->size(),
            moderate_slope_cloud_->size(), flat_terrain_cloud_rgb_->size());
    }

    // ============================================================
    // 步骤 9: 生成坡度可视化点云
    // ============================================================
    slope_cloud_->clear();
    for (int r = 0; r < slope_mat.rows; r++) {
        for (int c = 0; c < slope_mat.cols; c++) {
            if (valid_mask.at<uchar>(r, c) == 0) continue;

            int ind = terrain_height_grid_->Sub2Ind(c, r, 0);
            Eigen::Vector3d pos = terrain_height_grid_->Ind2Pos(ind);

            PCLPoint p;
            p.x = pos.x();
            p.y = pos.y();
            p.z = raw_h.at<float>(r, c);
            p.intensity = slope_mat.at<float>(r, c);

            slope_cloud_->push_back(p);
        }
    }

    slope_cloud_->width = slope_cloud_->size();
    slope_cloud_->height = 1;
    slope_cloud_->is_dense = false;
    slope_cloud_->header.frame_id = FARUtil::worldFrameId;
    slope_cloud_->header.stamp = pcl_conversions::toPCL(ros::Time::now());

    if (FARUtil::IsDebug) {
        ROS_INFO_THROTTLE(2.0, "MH: Slope cloud: %lu points", slope_cloud_->size());
    }
    // ============================================================
    // 步骤 10: 填充 terrain_grid_traverse_list_ 和 terrainHeightOut
    // ============================================================
    std::fill(terrain_grid_traverse_list_.begin(), terrain_grid_traverse_list_.end(), 0);
    terrainHeightOut->clear();

    // 遍历 ave_high_terrain_cloud_，标记可通行区域
    for (const auto& point : ave_high_terrain_cloud_->points) {
        Eigen::Vector3i sub =
            terrain_height_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, 0.0f));
        if (!terrain_height_grid_->InRange(sub)) continue;

        int r = sub.y();
        int c = sub.x();
        int ind = terrain_height_grid_->Sub2Ind(sub);

        // 只有非障碍区域才标记为可通行，并加入 terrainHeightOut
        if (obstacle_mask_.at<uchar>(r, c) == 0) {
            terrain_grid_traverse_list_[ind] = 1;
            terrainHeightOut->points.push_back(point);
        }
    }

    if (FARUtil::IsDebug) {
        ROS_INFO_THROTTLE(2.0, "MH: Traversable cells: %ld, terrainHeightOut: %lu points",
            std::count(terrain_grid_traverse_list_.begin(), terrain_grid_traverse_list_.end(), 1),
            terrainHeightOut->size());
    }
}

void MapHandler::PublishRiskMapViz() {
    risk_cloud_->clear();
    risk_cloud_rgb_->clear();
    // 遍历 risk_mask_mat_，把白色的点转回世界坐标
    for (int r = 0; r < risk_mask_mat_.rows; r++) {
        for (int c = 0; c < risk_mask_mat_.cols; c++) {
            int ind = terrain_height_grid_->Sub2Ind(c, r, 0);  // x=c, y=r
            Eigen::Vector3d pos = terrain_height_grid_->Ind2Pos(ind);
            float slope_val = slope_mat.at<float>(r, c);
            float step_val = inner_diff.at<float>(r, c);

            PointRGB p;
            p.x = pos.x();
            p.y = pos.y();
            p.z = pos.z();
            if (step_risk.at<uchar>(r, c) > 100) {
                p.r = 255;
                p.g = 0;
                p.b = 0;  // 红色  高度差
            } else if (occlusion_boundary_mask_.at<uchar>(r, c) > 100) {
                // 检查是否在地图边缘
                bool is_map_edge =
                    (r < 5 || r >= valid_mask.rows - 5 || c < 5 || c >= valid_mask.cols - 5);
                if (is_map_edge) {
                    p.r = 0;
                    p.g = 0;
                    p.b = 0;  // 灰色：地图边界
                } else {
                    p.r = 0;
                    p.g = 255;
                    p.b = 0;  // 绿色：真实遮挡
                }
            } else if (slope_risk.at<uchar>(r, c) > 100) {
                p.r = 255;
                p.g = 255;
                p.b = 0;  // 黄色  陡坡
            } else if (slope_val >= 0.176 && slope_val < 0.364 && step_val < 0.3) {
                p.r = 0;
                p.g = 0;
                p.b = 255;  // 蓝色  缓坡
            } else {
                p.r = 255;
                p.g = 255;
                p.b = 255;  // 白色  几乎视为平地
            }
            risk_cloud_rgb_->push_back(p);
        }
    }
}

PointCloudPtr MapHandler::GetRiskCloud() {
    return risk_cloud_;
}
PointCloudRGB MapHandler::GetRiskRBGCloud() {
    return risk_cloud_rgb_;
}
PointCloudPtr MapHandler::GetAveHeightCloud() {
    return ave_high_terrain_cloud_;
}
