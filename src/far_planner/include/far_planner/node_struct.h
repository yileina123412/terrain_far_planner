#ifndef NODE_STRUCT_H
#define NODE_STRUCT_H

#include "point_struct.h"

enum NodeType { NOT_DEFINED = 0, GROUND = 1, AIR = 2 };

enum NodeFreeDirect { UNKNOW = 0, CONVEX = 1, CONCAVE = 2, PILLAR = 3 };
enum TerrainType {
    TERRAIN_UNKNOWN = 0,
    TERRAIN_OBSTACLE = 1,
    TERRAIN_STEEP = 2,
    TERRAIN_MODERATE = 3,
    TERRAIN_FLAT = 4,
    TERRAIN_OCCLUSION = 5  //未知环境，前沿
};

typedef std::pair<Point3D, Point3D> PointPair;

namespace LiDARMODEL {
/* array resolution: 1 degree */
static const int kHorizontalFOV = 360;
static const int kVerticalFOV = 31;
static const float kAngleResX = 0.2;
static const float kAngleResY = 2.0;
}  // namespace LiDARMODEL

struct Polygon {
    Polygon() = default;
    std::size_t N;
    std::vector<Point3D> vertices;
    bool is_robot_inside;
    bool is_pillar;
    float perimeter;
};

typedef std::shared_ptr<Polygon> PolygonPtr;
typedef std::vector<PolygonPtr> PolygonStack;

struct CTNode {
    CTNode() = default;
    Point3D position;
    bool is_global_match;
    bool is_contour_necessary;
    bool is_ground_associate;
    std::size_t nav_node_id;
    NodeFreeDirect free_direct;

    PointPair surf_dirs;
    PolygonPtr poly_ptr;
    std::shared_ptr<CTNode> front;
    std::shared_ptr<CTNode> back;

    std::vector<std::shared_ptr<CTNode>> connect_nodes;

    // 1. 地形分类标签
    TerrainType terrain_type = TERRAIN_UNKNOWN;
    // 梯度信息
    Point3D gradient = Point3D(0.0f, 0.0f, 0.0f);
    float slop;
};

typedef std::shared_ptr<CTNode> CTNodePtr;
typedef std::vector<CTNodePtr> CTNodeStack;

struct NavNode {
    NavNode() = default;
    std::size_t id;
    Point3D position;
    PointPair surf_dirs;
    std::deque<Point3D> pos_filter_vec;
    std::deque<PointPair> surf_dirs_vec;
    CTNodePtr ctnode;
    bool is_active;
    bool is_block_frontier;
    bool is_contour_match;
    bool is_odom;
    bool is_goal;
    bool is_near_nodes;
    bool is_wide_near;
    bool is_merged;
    bool is_covered;
    bool is_frontier;
    bool is_finalized;
    bool is_navpoint;
    bool is_boundary;
    int clear_dumper_count;
    std::deque<int> frontier_votes;
    std::unordered_set<std::size_t> invalid_boundary;
    std::vector<std::shared_ptr<NavNode>> connect_nodes;
    std::vector<std::shared_ptr<NavNode>> poly_connects;
    std::vector<std::shared_ptr<NavNode>> contour_connects;
    std::unordered_map<std::size_t, std::deque<int>> contour_votes;
    std::unordered_map<std::size_t, std::deque<int>> edge_votes;
    std::vector<std::shared_ptr<NavNode>> potential_contours;
    std::vector<std::shared_ptr<NavNode>> potential_edges;
    std::vector<std::shared_ptr<NavNode>> trajectory_connects;
    std::unordered_map<std::size_t, std::size_t> trajectory_votes;
    std::unordered_map<std::size_t, std::size_t> terrain_votes;
    NodeType node_type;
    NodeFreeDirect free_direct;
    // planner members
    bool is_block_to_goal;
    bool is_traversable;
    bool is_free_traversable;
    float gscore, fgscore;
    std::shared_ptr<NavNode> parent;
    std::shared_ptr<NavNode> free_parent;

    // 地形感知属性
    // 1. 地形类型 (用于连接规则判断)
    TerrainType terrain_type = TERRAIN_UNKNOWN;
    // 如果三个都是 false，默认为 Flat (平地)
    // 2. 梯度向量 (用于 Cost 计算)
    // 存储归一化的梯度方向 (nx, ny, nz)
    Point3D gradient = Point3D(0.0f, 0.0f, 0.0f);
    float slop;
};

typedef std::shared_ptr<NavNode> NavNodePtr;
typedef std::pair<NavNodePtr, NavNodePtr> NavEdge;

struct nodeptr_equal {
    bool operator()(const NavNodePtr& n1, const NavNodePtr& n2) const {
        return n1->id == n2->id;
    }
};

struct navedge_hash {
    std::size_t operator()(const NavEdge& nav_edge) const {
        return boost::hash<std::pair<std::size_t, std::size_t>>()({nav_edge.first->id, nav_edge.second->id});
    }
};

struct nodeptr_hash {
    std::size_t operator()(const NavNodePtr& n_ptr) const {
        return std::hash<std::size_t>()(n_ptr->id);
    }
};

struct nodeptr_gcomp {
    bool operator()(const NavNodePtr& n1, const NavNodePtr& n2) const {
        return n1->gscore > n2->gscore;
    }
};

struct nodeptr_fgcomp {
    bool operator()(const NavNodePtr& n1, const NavNodePtr& n2) const {
        return n1->fgscore > n2->fgscore;
    }
};

struct nodeptr_icomp {
    bool operator()(const NavNodePtr& n1, const NavNodePtr& n2) const {
        return n1->position.intensity < n2->position.intensity;
    }
};

#endif
