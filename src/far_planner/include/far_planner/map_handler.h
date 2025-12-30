#ifndef MAP_HANDLER_H
#define MAP_HANDLER_H

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "utility.h"

enum CloudType { FREE_CLOUD = 0, OBS_CLOUD = 1 };

struct MapHandlerParams {
    MapHandlerParams() = default;
    float sensor_range;
    float floor_height;
    float cell_length;
    float cell_height;
    float grid_max_length;
    float grid_max_height;
    // local terrain height map
    float height_voxel_dim;
};

class MapHandler {
public:
    static std::unique_ptr<grid_ns::Grid<std::vector<float>>> terrain_height_grid_;
    // 记录地形点云每个格子平均高度
    static std::unique_ptr<grid_ns::Grid<float>> terrain_avg_height_grid_;
    PointCloudPtr ave_high_terrain_cloud_;
    static PointKdTreePtr kdtree_terrain_clould_;

    static cv::Mat grad_x, grad_y, slope_mat;
    // [新增] 五类地形的 CV 图
    static cv::Mat obstacle_mask_;            // 障碍物（红色）
    static cv::Mat occlusion_boundary_mask_;  // 遮挡边界（绿色）- 已有
    static cv::Mat steep_slope_mask_;         // 陡坡（黄色）
    static cv::Mat moderate_slope_mask_;      // 缓坡（蓝色）
    static cv::Mat flat_terrain_mask_;        // 平地（白色）

    // [新增] 五类地形的点云
    PointCloudRGB obstacle_cloud_;
    PointCloudRGB occlusion_cloud_;
    PointCloudRGB steep_slope_cloud_;
    PointCloudRGB moderate_slope_cloud_;
    PointCloudRGB flat_terrain_cloud_rgb_;

    PointCloudPtr slope_cloud_;  // 坡度可视化点云
    MapHandler() = default;
    ~MapHandler() = default;

    void Init(const MapHandlerParams& params);
    void SetMapOrigin(const Point3D& robot_pos);

    void UpdateRobotPosition(const Point3D& odom_pos);

    void AdjustNodesHeight(const NodePtrStack& nodes);

    void AdjustCTNodeHeight(const CTNodeStack& ctnodes);

    static float TerrainHeightOfPoint(const Point3D& p, bool& is_matched, const bool& is_search);

    static bool IsNavPointOnTerrainNeighbor(const Point3D& p, const bool& is_extend);

    static float NearestTerrainHeightofNavPoint(const Point3D& point, bool& is_associated);

    /**
     * @brief Calculate the terrain height of a given point and radius around it
     * @param p A given position
     * @param radius The radius distance around the given posiitn p
     * @param minH[out] The mininal terrain height in the radius
     * @param maxH[out] The maximal terrain height in the radius
     * @param is_match[out] Whether or not find terrain association in radius
     * @return The average terrain height
     * 在给定点 p 的二维平面半径 radius 范围内，利用 KdTree
     * 搜索所有地形平均高度点，返回这些点的平均高度，同时输出该范围内的最小高度、最大高度，并指示是否找到匹配点。
     */
    template <typename Position>
    static inline float NearestHeightOfRadius(
        const Position& p, const float& radius, float& minH, float& maxH, bool& is_matched) {
        std::vector<int> pIdxK;
        std::vector<float> pdDistK;
        PCLPoint pcl_p;
        pcl_p.x = p.x, pcl_p.y = p.y, pcl_p.z = 0.0f, pcl_p.intensity = 0.0f;
        minH = maxH = p.z;
        is_matched = false;
        if (kdtree_terrain_clould_->radiusSearch(pcl_p, radius, pIdxK, pdDistK) > 0) {
            float avgH = kdtree_terrain_clould_->getInputCloud()->points[pIdxK[0]].intensity;
            minH = maxH = avgH;
            for (int i = 1; i < pIdxK.size(); i++) {
                const float temp =
                    kdtree_terrain_clould_->getInputCloud()->points[pIdxK[i]].intensity;
                if (temp < minH) minH = temp;
                if (temp > maxH) maxH = temp;
                avgH += temp;
            }
            avgH /= (float)pIdxK.size();
            is_matched = true;
            return avgH;
        }
        return p.z;
    }

    /** Update global cloud grid with incoming clouds
     * @param CloudInOut incoming cloud ptr and output valid in range points
     */
    void UpdateObsCloudGrid(const PointCloudPtr& obsCloudInOut);
    void UpdateFreeCloudGrid(const PointCloudPtr& freeCloudIn);
    void UpdateTerrainHeightGrid(
        const PointCloudPtr& freeCloudIn, const PointCloudPtr& terrainHeightOut);

    void CalculateAveHigh();
    /** Extract Surrounding Free & Obs clouds
     * @param SurroundCloudOut output surrounding cloud ptr
     */
    void GetSurroundObsCloud(const PointCloudPtr& obsCloudOut);
    void GetSurroundFreeCloud(const PointCloudPtr& freeCloudOut);

    /** Extract Surrounding Free & Obs clouds
     * @param center the position of the grid that want to extract
     * @param cloudOut output cloud ptr
     * @param type choose free or obstacle cloud for extraction
     * @param is_large whether or not using the surrounding cells
     */
    void GetCloudOfPoint(const Point3D& center, const PointCloudPtr& CloudOut,
        const CloudType& type, const bool& is_large);

    /**
     * Get neihbor cells center positions
     * @param neighbor_centers[out] neighbor centers stack
     */
    void GetNeighborCeilsCenters(PointStack& neighbor_centers);

    /**
     * Get neihbor cells center positions
     * @param occupancy_centers[out] occupanied cells center stack
     */
    void GetOccupancyCeilsCenters(PointStack& occupancy_centers);

    /**
     * Remove pointcloud from grid map
     * @param obsCloud obstacle cloud points that need to be removed
     */
    void RemoveObsCloudFromGrid(const PointCloudPtr& obsCloud);

    /**
     * @brief Reset Current Grip Map Clouds
     */
    void ResetGripMapCloud();

    /**
     * @brief Clear the cells that from the robot position to the given position
     * @param point Give point location
     */
    void ClearObsCellThroughPosition(const Point3D& point);

    void GridToImg(cv::Mat& height_img, cv::Mat& var_img, cv::Mat& mask_img);

    void ComputeTerrainRiskAttributes(const PointCloudPtr& terrainHeightOut);

    void PublishRiskMapViz();

    PointCloudPtr GetRiskCloud();
    PointCloudRGB GetRiskRBGCloud();
    PointCloudPtr GetAveHeightCloud();

    // [新增] 获取五类地形的 CV 图
    cv::Mat GetObstacleMask() const {
        return obstacle_mask_;
    }
    cv::Mat GetOcclusionBoundaryMask() const {
        return occlusion_boundary_mask_;
    }
    cv::Mat GetSteepSlopeMask() const {
        return steep_slope_mask_;
    }
    cv::Mat GetModerateSlopeMask() const {
        return moderate_slope_mask_;
    }
    cv::Mat GetFlatTerrainMask() const {
        return flat_terrain_mask_;
    }

    // [新增] 获取五类地形的点云
    PointCloudRGB GetObstacleCloud() const {
        return obstacle_cloud_;
    }
    PointCloudRGB GetOcclusionCloud() const {
        return occlusion_cloud_;
    }
    PointCloudRGB GetSteepSlopeCloud() const {
        return steep_slope_cloud_;
    }
    PointCloudRGB GetModerateSlopeCloud() const {
        return moderate_slope_cloud_;
    }
    PointCloudRGB GetFlatTerrainCloudRGB() const {
        return flat_terrain_cloud_rgb_;
    }

    PointCloudPtr GetObsOutCloud() const {
        return obstacle_cloud_output_;
    }

    PointCloudPtr GetSteepOutCloud() const {
        return steep_slope_cloud_output_;
    }

    PointCloudPtr GetModerateOutCloud() const {
        return moderate_slope_cloud_output_;
    }

    PointCloudPtr GetSlopeCloud() const {
        return slope_cloud_;
    }
    cv::Mat GetGradX() const {
        return grad_x;
    }
    cv::Mat GetGradY() const {
        return grad_y;
    }

    // 根据世界坐标查询梯度向量
    static inline bool GetGradientAtPosition(
        const Point3D& world_pos, float& gx, float& gy, float& slope) {
        if (!terrain_height_grid_) return false;

        // 世界坐标 → 网格索引
        Eigen::Vector3i sub =
            terrain_height_grid_->Pos2Sub(Eigen::Vector3d(world_pos.x, world_pos.y, 0.0f));
        int r = sub.y();
        int c = sub.x();

        // 检查边界
        if (r < 0 || r >= grad_x.rows || c < 0 || c >= grad_x.cols) {
            return false;
        }

        // // 检查该点是否有效
        // if (valid_mask.at<uchar>(r, c) == 0) {
        //     return false;
        // }

        // 返回梯度
        gx = grad_x.at<float>(r, c);
        gy = grad_y.at<float>(r, c);
        slope = slope_mat.at<float>(r, c);
        return true;
    }
    // 根据位置获得当前位置的地形类型
    static inline TerrainType GetTerrainTypeAt(const Point3D& pos) {
        Eigen::Vector3i sub = terrain_height_grid_->Pos2Sub(Eigen::Vector3d(pos.x, pos.y, 0.0f));
        int r = sub.y();
        int c = sub.x();
        if (r < 0 || r >= obstacle_mask_.rows || c < 0 || c >= obstacle_mask_.cols)
            return TERRAIN_UNKNOWN;

        if (obstacle_mask_.at<uchar>(r, c) > 100) return TERRAIN_OBSTACLE;        // 障碍物
        if (steep_slope_mask_.at<uchar>(r, c) > 100) return TERRAIN_STEEP;        // 陡坡内部
        if (moderate_slope_mask_.at<uchar>(r, c) > 100) return TERRAIN_MODERATE;  // 缓坡
        if (flat_terrain_mask_.at<uchar>(r, c) > 100) return TERRAIN_FLAT;        // 平地
        // if (occlusion_boundary_mask_.at<uchar>(r, c) > 100) return TERRAIN_OCCLUSION;  //
        // 前沿区域

        // return TERRAIN_UNKNOWN;
        // 如果没有的话，默认是平地
        return TERRAIN_FLAT;
    }

private:
    MapHandlerParams map_params_;
    int neighbor_Lnum_, neighbor_Hnum_;
    Eigen::Vector3i robot_cell_sub_;
    int INFLATE_N;
    bool is_init_ = false;
    PointCloudPtr flat_terrain_cloud_;

    // 上帝视角的二值化风险地图
    bool occlusion_boundary_ready = false;
    bool risk_map_ready_ = false;
    Point3D initial_robot_pos_;

    cv::Mat risk_mask_mat_;
    // 高度 高度差
    cv::Mat raw_h, inner_diff, valid_mask;

    // 坡度风险，高度风险，最终风险
    cv::Mat slope_risk, step_risk, final_risk;
    PointCloudPtr risk_cloud_;
    PointCloudPtr obstacle_cloud_output_;
    PointCloudPtr steep_slope_cloud_output_, moderate_slope_cloud_output_;
    PointCloudRGB risk_cloud_rgb_;
    // ros::Publisher risk_debug_pub_;

    cv::Mat point_density_mat_;  // [新增] 点云密度图
    // cv::Mat occlusion_boundary_mask_;  // [新增] 遮挡边界掩膜
    // [新增] 滞后阈值所需的历史状态
    cv::Mat prev_steep_slope_mask_;
    cv::Mat prev_moderate_slope_mask_;
    bool is_first_classification_frame_;

    template <typename Position>
    static inline float NearestHeightOfPoint(const Position& p, float& dist_square) {
        // Find the nearest node in graph
        std::vector<int> pIdxK(1);
        std::vector<float> pdDistK(1);
        PCLPoint pcl_p;
        dist_square = FARUtil::kINF;
        pcl_p.x = p.x, pcl_p.y = p.y, pcl_p.z = 0.0f, pcl_p.intensity = 0.0f;
        if (kdtree_terrain_clould_->nearestKSearch(pcl_p, 1, pIdxK, pdDistK) > 0) {
            pcl_p = kdtree_terrain_clould_->getInputCloud()->points[pIdxK[0]];
            dist_square = pdDistK[0];
            return pcl_p.intensity;
        }
        return p.z;
    }

    void SetTerrainHeightGridOrigin(const Point3D& robot_pos);

    //
    inline void AssignFlatTerrainCloud(
        const PointCloudPtr& terrainRef, PointCloudPtr& terrainFlatOut) {
        const int N = terrainRef->size();
        terrainFlatOut->resize(N);
        for (int i = 0; i < N; i++) {
            PCLPoint pcl_p = terrainRef->points[i];
            pcl_p.intensity = pcl_p.z, pcl_p.z = 0.0f;
            terrainFlatOut->points[i] = pcl_p;
        }
    }

    inline void Expansion2D(
        const Eigen::Vector3i& csub, std::vector<Eigen::Vector3i>& subs, const int& n) {
        subs.clear();
        for (int ix = -n; ix <= n; ix++) {
            for (int iy = -n; iy <= n; iy++) {
                Eigen::Vector3i sub = csub;
                sub.x() += ix, sub.y() += iy;
                subs.push_back(sub);
            }
        }
    }

    void ObsNeighborCloudWithTerrain(
        std::unordered_set<int>& neighbor_obs, std::unordered_set<int>& extend_terrain_obs);
    // 这个邻居范围就是传感器半径，半径30m
    std::unordered_set<int> neighbor_free_indices_;  // surrounding free cloud grid indices stack
    static std::unordered_set<int>
        neighbor_obs_indices_;  // surrounding obs cloud grid indices stack
    static std::unordered_set<int>
        extend_obs_indices_;  // extended surrounding obs cloud grid indices stack

    std::vector<int> global_visited_induces_;
    std::vector<int> util_obs_modified_list_;
    std::vector<int> util_free_modified_list_;
    std::vector<int> util_remove_check_list_;
    static std::vector<int> terrain_grid_occupy_list_;
    static std::vector<int> terrain_grid_traverse_list_;

    static std::unique_ptr<grid_ns::Grid<PointCloudPtr>> world_free_cloud_grid_;
    static std::unique_ptr<grid_ns::Grid<PointCloudPtr>> world_obs_cloud_grid_;
};

#endif