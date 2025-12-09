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
    int height_dim = std::ceil((map_params_.sensor_range + map_params_.cell_length) * 2.0f / FARUtil::kLeafSize);
    if (height_dim % 2 == 0) height_dim++;
    Eigen::Vector3i height_grid_size(height_dim, height_dim, 1);
    Eigen::Vector3d height_grid_origin(0, 0, 0);
    // Eigen::Vector3d height_grid_resolution(FARUtil::robot_dim, FARUtil::robot_dim, FARUtil::kLeafSize);
    Eigen::Vector3d height_grid_resolution(FARUtil::kLeafSize, FARUtil::kLeafSize, FARUtil::kLeafSize);
    std::vector<float> temp_vec;
    terrain_height_grid_ = std::make_unique<grid_ns::Grid<std::vector<float>>>(
        height_grid_size, temp_vec, height_grid_origin, height_grid_resolution, 3);

    const int n_terrain_cell = terrain_height_grid_->GetCellNumber();
    terrain_grid_occupy_list_.resize(n_terrain_cell), terrain_grid_traverse_list_.resize(n_terrain_cell);
    std::fill(terrain_grid_occupy_list_.begin(), terrain_grid_occupy_list_.end(), 0);
    std::fill(terrain_grid_traverse_list_.begin(), terrain_grid_traverse_list_.end(), 0);

    INFLATE_N = 1;
    flat_terrain_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    risk_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    risk_cloud_rgb_ = PointCloudRGB(new pcl::PointCloud<pcl::PointXYZRGB>());
    ave_high_terrain_cloud_ = PointCloudPtr(new pcl::PointCloud<PCLPoint>());
    kdtree_terrain_clould_ = PointKdTreePtr(new pcl::KdTreeFLANN<PCLPoint>());
    kdtree_terrain_clould_->setSortedResults(false);

    // 风险评估
    risk_map_ready_ = false;
    initial_robot_pos_ = Point3D(0, 0, 0);  // 会在第一次更新时设置
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
            if (!world_obs_cloud_grid_->InRange(csub) || neighbor_obs_indices_.find(ind) == neighbor_obs_indices_.end())
                continue;
            world_obs_cloud_grid_->GetCell(ind)->clear();
            if (world_free_cloud_grid_->GetCell(ind)->empty()) {
                global_visited_induces_[ind] = 0;
            }
        }
    }
}

void MapHandler::GetCloudOfPoint(
    const Point3D& center, const PointCloudPtr& cloudOut, const CloudType& type, const bool& is_large) {
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
    map_origin.z =
        ori_robot_pos.z - (map_params_.cell_height * dim.z()) / 2.0f - FARUtil::vehicle_height;  // From Ground Level
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
    robot_cell_sub_ = world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(odom_pos.x, odom_pos.y, odom_pos.z));
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
            for (int k = -H; k <= H; k++) {
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
        Eigen::Vector3i sub = world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
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
        Eigen::Vector3i sub = world_free_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
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

float MapHandler::TerrainHeightOfPoint(const Point3D& p, bool& is_matched, const bool& is_search) {
    is_matched = false;
    const Eigen::Vector3i sub = terrain_height_grid_->Pos2Sub(Eigen::Vector3d(p.x, p.y, 0.0f));
    if (terrain_height_grid_->InRange(sub)) {
        const int ind = terrain_height_grid_->Sub2Ind(sub);
        if (terrain_grid_traverse_list_[ind] != 0) {
            is_matched = true;
            return terrain_height_grid_->GetCell(ind)[0];
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
    const Eigen::Vector3i ori_sub = world_free_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, p_th));
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

bool MapHandler::IsNavPointOnTerrainNeighbor(const Point3D& point, const bool& is_extend) {
    const float h = point.z - FARUtil::vehicle_height;
    const Eigen::Vector3i sub = world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, h));
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

void MapHandler::AdjustNodesHeight(const NodePtrStack& nodes) {
    if (nodes.empty()) return;
    for (const auto& node_ptr : nodes) {
        if (!node_ptr->is_active || node_ptr->is_boundary || FARUtil::IsFreeNavNode(node_ptr) ||
            FARUtil::IsOutsideGoal(node_ptr) || !FARUtil::IsPointInLocalRange(node_ptr->position, true)) {
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
        const float avg_h = NearestHeightOfRadius(
            ctnode_ptr->position, FARUtil::kMatchDist, min_th, max_th, ctnode_ptr->is_ground_associate);
        if (ctnode_ptr->is_ground_associate) {
            ctnode_ptr->position.z = min_th + FARUtil::vehicle_height;
            ctnode_ptr->position.z = std::max(std::min(ctnode_ptr->position.z, H_MAX), H_MIN);
        } else {
            ctnode_ptr->position.z = TerrainHeightOfPoint(ctnode_ptr->position, ctnode_ptr->is_ground_associate, true);
            ctnode_ptr->position.z += FARUtil::vehicle_height;
            ctnode_ptr->position.z = std::max(std::min(ctnode_ptr->position.z, H_MAX), H_MIN);
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
        const Eigen::Vector3i sub = terrain_height_grid_->Pos2Sub(Eigen::Vector3d(pos.x, pos.y, 0.0f));
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

void MapHandler::UpdateTerrainHeightGrid(const PointCloudPtr& freeCloudIn, const PointCloudPtr& terrainHeightOut) {
    if (freeCloudIn->empty()) return;
    PointCloudPtr copy_free_ptr(new pcl::PointCloud<PCLPoint>());
    pcl::copyPointCloud(*freeCloudIn, *copy_free_ptr);
    FARUtil::FilterCloud(copy_free_ptr, terrain_height_grid_->GetResolution());
    std::fill(terrain_grid_occupy_list_.begin(), terrain_grid_occupy_list_.end(), 0);
    for (const auto& point : copy_free_ptr->points) {
        Eigen::Vector3i csub = terrain_height_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, 0.0f));
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
    ComputeTerrainRiskAttributes();
    this->TraversableAnalysis(terrainHeightOut);
    terrainHeightOut->header.frame_id = FARUtil::worldFrameId;  // 设置为世界坐标系
    terrainHeightOut->header.stamp = pcl_conversions::toPCL(ros::Time::now());
    if (terrainHeightOut->empty()) {  // set terrain height kdtree
        FARUtil::ClearKdTree(flat_terrain_cloud_, kdtree_terrain_clould_);
    } else {
        this->AssignFlatTerrainCloud(terrainHeightOut, flat_terrain_cloud_);
        kdtree_terrain_clould_->setInputCloud(flat_terrain_cloud_);
    }
    // update surrounding obs cloud grid indices based on terrain
    this->ObsNeighborCloudWithTerrain(neighbor_obs_indices_, extend_obs_indices_);
}

void MapHandler::CalculateAveHigh() {
    const Eigen::Vector3i robot_sub =
        terrain_height_grid_->Pos2Sub(Eigen::Vector3d(FARUtil::robot_pos.x, FARUtil::robot_pos.y, 0.0f));
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
        if (height_vec.empty()) continue;

        // 计算该网格中所有高度的平均值
        float avg_height = 0.0f;
        for (const float& height : height_vec) {
            avg_height += height;
        }
        avg_height /= height_vec.size();

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
    }
    ave_high_terrain_cloud_->width = ave_high_terrain_cloud_->points.size();
    ave_high_terrain_cloud_->height = 1;
    ave_high_terrain_cloud_->is_dense = false;
    ave_high_terrain_cloud_->header.frame_id = FARUtil::worldFrameId;
    ave_high_terrain_cloud_->header.stamp = pcl_conversions::toPCL(ros::Time::now());
}

void MapHandler::TraversableAnalysis(const PointCloudPtr& terrainHeightOut) {
    const Eigen::Vector3i robot_sub =
        terrain_height_grid_->Pos2Sub(Eigen::Vector3d(FARUtil::robot_pos.x, FARUtil::robot_pos.y, 0.0f));
    terrainHeightOut->clear();
    if (!terrain_height_grid_->InRange(robot_sub)) {
        ROS_ERROR("MH: terrain height analysis error: robot position is not in range");
        return;
    }
    const float H_THRED = map_params_.height_voxel_dim;
    std::fill(terrain_grid_traverse_list_.begin(), terrain_grid_traverse_list_.end(), 0);
    // Lambda Function
    auto IsTraversableNeighbor = [&](const int& cur_id, const int& ref_id) {
        if (terrain_grid_occupy_list_[ref_id] == 0) return false;
        const float cur_h = terrain_height_grid_->GetCell(cur_id)[0];
        float ref_h = 0.0f;
        int counter = 0;
        for (const auto& e : terrain_height_grid_->GetCell(ref_id)) {
            if (abs(e - cur_h) > H_THRED) continue;
            ref_h += e, counter++;
        }
        if (counter > 0) {
            terrain_height_grid_->GetCell(ref_id).resize(1);
            terrain_height_grid_->GetCell(ref_id)[0] = ref_h / (float)counter;
            return true;
        }
        return false;
    };

    auto AddTraversePoint = [&](const int& idx) {
        Eigen::Vector3d cpos = terrain_height_grid_->Ind2Pos(idx);
        cpos.z() = terrain_height_grid_->GetCell(idx)[0];
        const PCLPoint p = FARUtil::Point3DToPCLPoint(Point3D(cpos));
        terrainHeightOut->points.push_back(p);
        terrain_grid_traverse_list_[idx] = 1;
    };

    const int robot_idx = terrain_height_grid_->Sub2Ind(robot_sub);
    const std::array<int, 4> dx = {-1, 0, 1, 0};
    const std::array<int, 4> dy = {0, 1, 0, -1};
    std::deque<int> q;
    bool is_robot_terrain_init = false;
    std::unordered_set<int> visited_set;
    q.push_back(robot_idx), visited_set.insert(robot_idx);
    while (!q.empty()) {
        const int cur_id = q.front();
        q.pop_front();
        if (terrain_grid_occupy_list_[cur_id] != 0) {
            if (!is_robot_terrain_init) {
                float avg_h = 0.0f;
                int counter = 0;
                for (const auto& e : terrain_height_grid_->GetCell(cur_id)) {
                    if (abs(e - FARUtil::robot_pos.z + FARUtil::vehicle_height) > H_THRED) continue;
                    avg_h += e, counter++;
                }
                if (counter > 0) {
                    avg_h /= (float)counter;
                    terrain_height_grid_->GetCell(cur_id).resize(1);
                    terrain_height_grid_->GetCell(cur_id)[0] = avg_h;
                    AddTraversePoint(cur_id);
                    is_robot_terrain_init = true;  // init terrain height map current robot height
                    q.clear();
                }
            } else {
                AddTraversePoint(cur_id);
            }
        } else if (is_robot_terrain_init) {
            continue;
        }
        const Eigen::Vector3i csub = terrain_height_grid_->Ind2Sub(cur_id);
        for (int i = 0; i < 4; i++) {
            Eigen::Vector3i ref_sub = csub;
            ref_sub.x() += dx[i], ref_sub.y() += dy[i];
            if (!terrain_height_grid_->InRange(ref_sub)) continue;
            const int ref_id = terrain_height_grid_->Sub2Ind(ref_sub);
            if (!visited_set.count(ref_id) && (!is_robot_terrain_init || IsTraversableNeighbor(cur_id, ref_id))) {
                q.push_back(ref_id);
                visited_set.insert(ref_id);
            }
        }
    }
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
        Eigen::Vector3i sub = world_obs_cloud_grid_->Pos2Sub(Eigen::Vector3d(point.x, point.y, point.z));
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

void MapHandler::GridToImg(const std::unique_ptr<grid_ns::Grid<std::vector<float>>>& grid, cv::Mat& height_img,
    cv::Mat& var_img, cv::Mat& mask_img) {
    const Eigen::Vector3i dim = grid->GetSize();
    height_img = cv::Mat::zeros(dim.y(), dim.x(), CV_32FC1);
    var_img = cv::Mat::zeros(dim.y(), dim.x(), CV_32FC1);
    mask_img = cv::Mat::zeros(dim.y(), dim.x(), CV_8UC1);

    for (int i = 0; i < grid->GetCellNumber(); ++i) {
        if (terrain_grid_occupy_list_[i] == 0) continue;
        // 获取行列坐标 (注意 OpenCV 行列与 Grid x,y 的对应)
        Eigen::Vector3i sub = grid->Ind2Sub(i);
        int r = sub.y();  // row -> y
        int c = sub.x();  // col -> x

        const auto& points = grid->GetCell(i);
        if (!points.empty()) {
            float min_z = 10000, max_z = -10000, sum = 0;
            for (float z : points) {
                if (z < min_z) min_z = z;
                if (z > max_z) max_z = z;
                sum += z;
            }
            height_img.at<float>(r, c) = sum / points.size();  // 平均高度
            var_img.at<float>(r, c) = max_z - min_z;           // 格内落差
            mask_img.at<uchar>(r, c) = 255;                    // 有效标记
        }
    }
}

void MapHandler::ComputeTerrainRiskAttributes() {
    if (!terrain_height_grid_) return;

    // inner_diff:悬崖图
    cv::Mat raw_h, inner_diff, valid_mask;
    GridToImg(terrain_height_grid_, raw_h, inner_diff, valid_mask);

    // 1. 填补空洞 (形态学闭运算)
    // 解决车底黑洞问题，但不乱补悬崖外
    // 目的：把车底这种被包围的小黑洞填上，但不要把悬崖外面的大黑洞填上。
    cv::Mat closed_mask;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // 执行闭运算：先膨胀再腐蚀
    cv::morphologyEx(valid_mask, closed_mask, cv::MORPH_CLOSE, kernel);

    // 2. 生成平滑后的高度图 (用于算坡度)
    // 我们需要一张“连续”的高度图来算坡度，不能有断层。
    cv::Mat filled_h = raw_h.clone();

    // 简单的均值填补: 遍历闭运算补出来的区域
    for (int r = 1; r < raw_h.rows - 1; r++) {
        for (int c = 1; c < raw_h.cols - 1; c++) {
            if (valid_mask.at<unsigned char>(r, c) == 0 && closed_mask.at<unsigned char>(r, c) == 255) {
                float sum = 0.0f;
                int count = 0;

                // 遍历 3x3 邻域
                for (int nr = -1; nr <= 1; ++nr) {
                    for (int nc = -1; nc <= 1; ++nc) {
                        // 边界检查 (虽然外层循环已避开边界，但为了通用性加上)
                        if (r + nr < 0 || r + nr >= raw_h.rows || c + nc < 0 || c + nc >= raw_h.cols) continue;

                        // 只统计原始数据中“有效”的点
                        if (valid_mask.at<unsigned char>(r + nr, c + nc) == 255) {
                            sum += raw_h.at<float>(r + nr, c + nc);
                            count++;
                        }
                    }
                }

                if (count > 0) {
                    // 用周围有效点的平均值填补
                    filled_h.at<float>(r, c) = sum / (float)count;
                } else {
                    // 如果周围一圈也没数据（极少见，除非是孤立噪声点），回退到继承左边的点
                    filled_h.at<float>(r, c) = filled_h.at<float>(r, c - 1);
                }
            }
        }
    }

    // 3. 计算坡度 (Slope)  坡度图
    cv::Mat grad_x, grad_y, slope_mat;
    float res = terrain_height_grid_->GetResolution().x();
    cv::Sobel(filled_h, grad_x, CV_32F, 1, 0, 3, 1.0 / (8.0 * res));
    cv::Sobel(filled_h, grad_y, CV_32F, 0, 1, 3, 1.0 / (8.0 * res));
    cv::magnitude(grad_x, grad_y, slope_mat);  // 计算每个像素点的模长

    // ------------------------------------------------------------------------
    // [新增 2]：距离置信度衰减 (解决远处的假阳性/黄色闪烁)
    // ------------------------------------------------------------------------
    // 假设 grid 是以机器人为中心的 (local map)，那么图像中心就是机器人位置
    int center_r = raw_h.rows / 2;
    int center_c = raw_h.cols / 2;
    float max_trust_dist = 15.0;    // 15米内完全信任
    float decay_start_dist = 10.0;  // 10米开始衰减
    for (int r = 0; r < slope_mat.rows; r++) {
        for (int c = 0; c < slope_mat.cols; c++) {
            // 计算物理距离 (米)
            float dist = sqrt(pow(r - center_r, 2) + pow(c - center_c, 2)) * res;

            // 2. 远处坡度衰减
            if (dist > decay_start_dist) {
                float decay = 1.0f;
                if (dist >= max_trust_dist) {
                    decay = 0.1f;  // 极远处几乎不信
                } else {
                    // 线性衰减
                    decay = 1.0f - (dist - decay_start_dist) / (max_trust_dist - decay_start_dist);
                    if (decay < 0.1f) decay = 0.1f;
                }
                // 压低远处的坡度值，让它变回蓝色或白色
                slope_mat.at<float>(r, c) *= decay;
            }
        }
    }

    // 4. 生成 Risk Mask
    cv::Mat slope_risk, step_risk, final_risk;
    // 阈值：坡度 > 0.5 (约26度)，格内落差 > 0.3m
    cv::threshold(slope_mat, slope_risk, 0.5, 255, cv::THRESH_BINARY);
    cv::threshold(inner_diff, step_risk, 0.3, 255, cv::THRESH_BINARY);

    // [新增] 去除红色零星噪点
    // 定义一个很小的核，比如 3x3
    cv::Mat remove_noise_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // 执行开运算：消除孤立点
    cv::morphologyEx(step_risk, step_risk, cv::MORPH_OPEN, remove_noise_kernel);

    slope_risk.convertTo(slope_risk, CV_8UC1);
    step_risk.convertTo(step_risk, CV_8UC1);

    // [修改] 检查是否需要屏蔽陡坡风险（黄色）
    if (!risk_map_ready_) {
        float dist_moved = sqrt(
            pow(FARUtil::robot_pos.x - initial_robot_pos_.x, 2) + pow(FARUtil::robot_pos.y - initial_robot_pos_.y, 2));
        if (dist_moved < 1.0) {
            // 只清除机器人周围1.5米半径的黄色陡坡风险
            float clear_radius = 1.5;  // 清除半径1.5米
            for (int r = 0; r < slope_risk.rows; r++) {
                for (int c = 0; c < slope_risk.cols; c++) {
                    float dist = sqrt(pow(r - center_r, 2) + pow(c - center_c, 2)) * res;
                    if (dist < clear_radius) {
                        slope_risk.at<uchar>(r, c) = 0;  // 只清零半径内的陡坡风险
                    }
                }
            }
            if (FARUtil::IsDebug) {
                ROS_INFO_THROTTLE(1.0, "MH: Slope risk within %.1fm suppressed, moved %.2fm", clear_radius, dist_moved);
            }
        } else {
            risk_map_ready_ = true;
            if (FARUtil::IsDebug) {
                ROS_INFO("MH: Risk map fully activated after moving %.2f meters.", dist_moved);
            }
        }
    }

    cv::bitwise_or(slope_risk, step_risk, final_risk);

    // 5. [最关键一步] 过滤边界噪声
    // 只保留“被有效区域包围”的风险，去掉地图边缘的风险
    cv::Mat safe_zone;
    cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::erode(valid_mask, safe_zone, erode_kernel);          // 腐蚀有效区域
    cv::bitwise_and(final_risk, safe_zone, risk_mask_mat_);  // 赋值给成员变量

    // **DEBUG**: 发布出去看看！
    PublishRiskMapViz();
    risk_cloud_->clear();
    risk_cloud_rgb_->clear();
    // 遍历 risk_mask_mat_，把白色的点转回世界坐标
    for (int r = 0; r < risk_mask_mat_.rows; r++) {
        for (int c = 0; c < risk_mask_mat_.cols; c++) {
            if (risk_mask_mat_.at<uchar>(r, c) > 100) {  // 是障碍
                // Image -> Grid Index
                int ind = terrain_height_grid_->Sub2Ind(c, r, 0);  // x=c, y=r
                Eigen::Vector3d pos = terrain_height_grid_->Ind2Pos(ind);

                PCLPoint p;
                p.x = pos.x();
                p.y = pos.y();
                p.z = pos.z();  // 这里可以用地形高度，或者为了明显设高一点
                // 判断风险类型
                if (step_risk.at<uchar>(r, c) > 100) {
                    p.intensity = 200;  // 台阶/落差型风险
                } else if (slope_risk.at<uchar>(r, c) > 100) {
                    p.intensity = 100;  // 陡坡型风险 红色
                } else {
                    p.intensity = 50;  // 其他
                }
                risk_cloud_->push_back(p);
            }

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

            // // 新增：显示坡度在10-20度且落差小于0.3的格子
            // float slope_val = slope_mat.at<float>(r, c);
            // float step_val = inner_diff.at<float>(r, c);
            // if (step_val < 0.3 && slope_val >= 0.176 && slope_val < 0.364) {
            //     int ind = terrain_height_grid_->Sub2Ind(c, r, 0);
            //     Eigen::Vector3d pos = terrain_height_grid_->Ind2Pos(ind);

            //     PCLPoint p;
            //     p.x = pos.x();
            //     p.y = pos.y();
            //     p.z = pos.z();
            //     p.intensity = 80;  // 你可以选一个独特的值，比如80，代表10-20度坡度

            //     risk_cloud_->push_back(p);
            // }
        }
    }
}

void MapHandler::PublishRiskMapViz() {}

PointCloudPtr MapHandler::GetRiskCloud() {
    return risk_cloud_;
}
PointCloudRGB MapHandler::GetRiskRBGCloud() {
    return risk_cloud_rgb_;
}