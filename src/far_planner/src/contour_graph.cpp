/*
 * FAR Planner
 * Copyright (C) 2021 Fan Yang - All rights reserved
 * fanyang2@andrew.cmu.edu,
 */

#include "far_planner/contour_graph.h"

#include "far_planner/intersection.h"

/***************************************************************************************/

void ContourGraph::Init(const ContourGraphParams& params) {
    ctgraph_params_ = params;
    ContourGraph::contour_graph_.clear();
    ContourGraph::contour_polygons_.clear();
    ContourGraph::steep_polygons_.clear();
    ALIGN_ANGLE_COS = cos(M_PI - FARUtil::kAcceptAlign / 2.0f);
    is_robot_inside_poly_ = false;
    ContourGraph::global_contour_set_.clear();
    ContourGraph::boundary_contour_set_.clear();
}
// 传入轮廓节点，将轮廓点转化为ct点，并整理多边形
void ContourGraph::UpdateContourGraph(
    const NavNodePtr& odom_node_ptr, const std::vector<std::vector<Point3D>>& filtered_contours) {
    odom_node_ptr_ = odom_node_ptr;
    this->ClearContourGraph();
    // 创建多边形
    for (const auto& poly : filtered_contours) {
        PolygonPtr new_poly_ptr = NULL;
        this->CreatePolygon(poly, new_poly_ptr);
        this->AddPolyToContourPolygon(new_poly_ptr);
    }
    ContourGraph::UpdateOdomFreePosition(odom_node_ptr_, FARUtil::free_odom_p);
    // 遍历多边形并创建ct点
    for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
        poly_ptr->is_robot_inside =
            FARUtil::PointInsideAPoly(poly_ptr->vertices, FARUtil::free_odom_p);
        CTNodePtr new_ctnode_ptr = NULL;
        if (poly_ptr->is_pillar) {
            Point3D mean_p = FARUtil::AveragePoints(poly_ptr->vertices);
            this->CreateCTNodeObs(mean_p, new_ctnode_ptr, poly_ptr, true);
            this->AddCTNodeToGraph(new_ctnode_ptr);
        } else {
            CTNodeStack ctnode_stack;
            ctnode_stack.clear();
            const int N = poly_ptr->vertices.size();
            for (std::size_t idx = 0; idx < N; idx++) {
                this->CreateCTNodeObs(poly_ptr->vertices[idx], new_ctnode_ptr, poly_ptr, false);
                ctnode_stack.push_back(new_ctnode_ptr);
            }
            // add connections to contour nodes 把前后节点补上
            for (int idx = 0; idx < N; idx++) {
                int ref_idx = FARUtil::Mod(idx - 1, N);
                ctnode_stack[idx]->front = ctnode_stack[ref_idx];
                ref_idx = FARUtil::Mod(idx + 1, N);
                ctnode_stack[idx]->back = ctnode_stack[ref_idx];
                this->AddCTNodeToGraph(ctnode_stack[idx]);
            }
            // add first ctnode of each polygon to poly ctnodes stack
            if (!ctnode_stack.empty()) ContourGraph::polys_ctnodes_.push_back(ctnode_stack.front());
        }
    }
    this->AnalysisSurfAngleAndConvexity(ContourGraph::contour_graph_);
}
void ContourGraph::UpdateContourGraphTerrain(
    const std::vector<std::vector<Point3D>>& steep_boundary,
    const std::vector<std::vector<Point3D>>& steep_inner,
    const std::vector<std::vector<Point3D>>& moderate_boundary,
    const std::vector<std::vector<Point3D>>& moderate_inner) {
    // [新增] 前置检查
    if (!MapHandler::terrain_height_grid_ || MapHandler::grad_x.empty() ||
        MapHandler::grad_y.empty()) {
        ROS_WARN("CG: Terrain maps not ready, skipping UpdateContourGraphTerrain");
        return;
    }

    // 处理陡坡边界点和内部点 (一一对应)
    for (size_t cluster_idx = 0; cluster_idx < steep_boundary.size(); cluster_idx++) {
        const auto& boundary_cluster = steep_boundary[cluster_idx];

        if (boundary_cluster.size() < 3) {
            ROS_WARN("CG: Steep boundary cluster %lu too small (%lu points), skipping", cluster_idx,
                boundary_cluster.size());
            continue;
        }

        // 1. 创建多边形并添加到 contour_polygons_
        PolygonPtr steep_poly_ptr = NULL;
        this->CreatePolygon(boundary_cluster, steep_poly_ptr);
        steep_poly_ptr->is_robot_inside = false;  // 陡坡多边形机器人不在内部
        this->AddPolyToSteepPolygon(steep_poly_ptr);

        // 2. 为边界点创建 CTNode (带 front/back 连接)
        CTNodeStack ctnode_stack;
        ctnode_stack.clear();

        for (const auto& point : boundary_cluster) {
            CTNodePtr new_ctnode_ptr = NULL;
            this->CreatCTNodeTerrain(point, new_ctnode_ptr, true, TerrainType::TERRAIN_STEEP_BOUND);
            new_ctnode_ptr->poly_ptr = steep_poly_ptr;  // 关联多边形
            ctnode_stack.push_back(new_ctnode_ptr);
        }

        // 连接 front 和 back (形成闭环轮廓)
        const int N = ctnode_stack.size();
        for (int idx = 0; idx < N; idx++) {
            int ref_idx = FARUtil::Mod(idx - 1, N);
            ctnode_stack[idx]->front = ctnode_stack[ref_idx];
            ref_idx = FARUtil::Mod(idx + 1, N);
            ctnode_stack[idx]->back = ctnode_stack[ref_idx];
            this->AddCTNodeToGraph(ctnode_stack[idx]);
        }

        // 添加首个边界节点到 polys_ctnodes_
        if (!ctnode_stack.empty()) {
            ContourGraph::polys_ctnodes_.push_back(ctnode_stack.front());
        }

        // 3. 处理对应的内部点 (共用同一个多边形)
        if (cluster_idx < steep_inner.size()) {
            const auto& inner_cluster = steep_inner[cluster_idx];
            for (const auto& point : inner_cluster) {
                CTNodePtr new_ctnode_ptr = NULL;
                this->CreatCTNodeTerrain(point, new_ctnode_ptr, true, TerrainType::TERRAIN_STEEP);
                new_ctnode_ptr->poly_ptr = steep_poly_ptr;  // 关联到同一个多边形
                this->AddCTNodeToGraph(new_ctnode_ptr);
            }
            ROS_INFO("CG: Added steep cluster %lu: %d boundary + %lu inner CTNodes", cluster_idx, N,
                inner_cluster.size());
        } else {
            ROS_INFO("CG: Added steep cluster %lu: %d boundary CTNodes (no inner)", cluster_idx, N);
        }
    }

    // 处理缓坡内部点
    for (const auto& inner_cluster : moderate_inner) {
        for (const auto& point : inner_cluster) {
            CTNodePtr new_ctnode_ptr = NULL;
            this->CreatCTNodeTerrain(point, new_ctnode_ptr, true, TerrainType::TERRAIN_MODERATE);
            this->AddCTNodeToGraph(new_ctnode_ptr);
        }
        ROS_INFO("CG: Added %lu moderate inner CTNodes", inner_cluster.size());
    }
}

/* 将ct点匹配导航点 */
void ContourGraph::MatchContourWithNavGraph(const NodePtrStack& global_nodes,
    const NodePtrStack& near_nodes, CTNodeStack& new_convex_vertices) {
    // 清空匹配联系
    for (const auto& node_ptr : global_nodes) {
        node_ptr->is_contour_match = false;
        node_ptr->ctnode = NULL;
    }
    // 遍历当前的ct点
    for (const auto& ctnode_ptr : ContourGraph::contour_graph_) {  // distance match
        ctnode_ptr->is_global_match = false;
        ctnode_ptr->nav_node_id = 0;
        if (ctnode_ptr->free_direct != NodeFreeDirect::UNKNOW) {
            const NavNodePtr matched_node = this->NearestNavNodeForCTNode(ctnode_ptr, near_nodes);
            // 验证匹配的有效性
            if (matched_node != NULL && IsCTMatchLineFreePolygon(ctnode_ptr, matched_node, false)) {
                this->MatchCTNodeWithNavNode(ctnode_ptr, matched_node);
            }
        }
    }
    // 检查匹配到导航点的ct点的完整性和合理性，防止过渡简化导致轮廓缺失
    this->EnclosePolygonsCheck();
    // 处理未匹配到的ct点
    new_convex_vertices.clear();
    for (const auto& ctnode_ptr : ContourGraph::contour_graph_) {  // Get new vertices
        if (!ctnode_ptr->is_global_match && ctnode_ptr->free_direct != NodeFreeDirect::UNKNOW) {
            if (ctnode_ptr->free_direct != NodeFreeDirect::PILLAR) {  // check wall contour
                const float dot_value = ctnode_ptr->surf_dirs.first * ctnode_ptr->surf_dirs.second;
                if (dot_value < ALIGN_ANGLE_COS) continue;  // wall detected
            }
            new_convex_vertices.push_back(ctnode_ptr);
        }
    }
}
// 多边形碰撞检测  确保连接线段在自由空间中（不与任何障碍物多边形碰撞）
bool ContourGraph::IsNavNodesConnectFreePolygon(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    if (node_ptr1->is_navpoint || node_ptr2->is_navpoint) {
        if ((node_ptr1->position - node_ptr2->position).norm() <
            FARUtil::kNavClearDist) {  // connect to internav node
            return true;
        }
    }
    // ========== 2. 确定检查范围（局部/全局）==========
    const bool is_global_check =
        ContourGraph::IsNeedGlobalCheck(node_ptr1->position, node_ptr2->position);
    // ========== 3. 边缘重投影（提高碰撞检测精度）==========  0.2m
    // 原始连接线可能穿过轮廓节点的"表面方向"，导致误判为碰撞
    ConnectPair cedge =
        ContourGraph::ReprojectEdge(node_ptr1, node_ptr2, FARUtil::kProjectDist, is_global_check);
    if (node_ptr1->is_odom) {
        cedge.start_p = cv::Point2f(FARUtil::free_odom_p.x, FARUtil::free_odom_p.y);
    } else if (node_ptr2->is_odom) {
        cedge.end_p = cv::Point2f(FARUtil::free_odom_p.x, FARUtil::free_odom_p.y);
    }
    ConnectPair bd_cedge = cedge;
    const HeightPair h_pair(node_ptr1->position, node_ptr2->position);
    if (!node_ptr1->is_boundary)
        bd_cedge.start_p = cv::Point2f(node_ptr1->position.x, node_ptr1->position.y);
    if (!node_ptr2->is_boundary)
        bd_cedge.end_p = cv::Point2f(node_ptr2->position.x, node_ptr2->position.y);
    return ContourGraph::IsPointsConnectFreePolygon(cedge, bd_cedge, h_pair, is_global_check);
}

bool ContourGraph::IsPoint3DConnectFreePolygon(const Point3D& p1, const Point3D& p2) {
    const bool is_global_check = ContourGraph::IsNeedGlobalCheck(p1, p2);
    const ConnectPair ori_cedge(p1, p2);
    const ConnectPair cedge = ori_cedge;
    const HeightPair h_pair(p1, p2);
    return ContourGraph::IsPointsConnectFreePolygon(cedge, ori_cedge, h_pair, is_global_check);
}

bool ContourGraph::IsEdgeCollideBoundary(const Point3D& p1, const Point3D& p2) {
    if (ContourGraph::boundary_contour_.empty()) return false;
    const ConnectPair edge = ConnectPair(p1, p2);
    for (const auto& contour : ContourGraph::boundary_contour_) {
        if (ContourGraph::IsEdgeCollideSegment(contour, edge)) {
            return true;
        }
    }
    return false;
}

bool ContourGraph::IsNavToGoalConnectFreePolygon(
    const NavNodePtr& node_ptr, const NavNodePtr& goal_ptr) {
    if ((node_ptr->position - goal_ptr->position).norm() < FARUtil::kNavClearDist) return true;
    HeightPair h_pair(node_ptr->position, goal_ptr->position);
    bool is_global_check = ContourGraph::IsNeedGlobalCheck(node_ptr->position, goal_ptr->position);
    if (FARUtil::IsMultiLayer) {
        if (!FARUtil::IsAtSameLayer(node_ptr, goal_ptr) && !node_ptr->is_frontier) return false;
        h_pair = HeightPair(goal_ptr->position.z, goal_ptr->position.z);
        is_global_check = true;
    }
    const ConnectPair cedge =
        ContourGraph::ReprojectEdge(node_ptr, goal_ptr, FARUtil::kProjectDist, is_global_check);
    ConnectPair bd_cedge = cedge;
    if (!node_ptr->is_boundary)
        bd_cedge.start_p = cv::Point2f(node_ptr->position.x, node_ptr->position.y);
    if (!goal_ptr->is_boundary)
        bd_cedge.end_p = cv::Point2f(goal_ptr->position.x, goal_ptr->position.y);
    return ContourGraph::IsPointsConnectFreePolygon(cedge, bd_cedge, h_pair, is_global_check);
}

bool ContourGraph::IsCTMatchLineFreePolygon(const CTNodePtr& matched_ctnode,
    const NavNodePtr& matched_navnode, const bool& is_global_check) {
    if ((matched_ctnode->position - matched_navnode->position).norm() < FARUtil::kNavClearDist)
        return true;
    const HeightPair h_pair(matched_ctnode->position, matched_navnode->position);
    const ConnectPair bd_cedge = ConnectPair(matched_ctnode->position, matched_navnode->position);
    const ConnectPair cedge =
        ContourGraph::ReprojectEdge(matched_ctnode, matched_navnode, FARUtil::kProjectDist);
    return ContourGraph::IsPointsConnectFreePolygon(cedge, bd_cedge, h_pair, is_global_check);
}
// 检查两点之间的连接线是否和多边形发生碰撞
bool ContourGraph::IsPointsConnectFreePolygon(const ConnectPair& cedge, const ConnectPair& bd_cedge,
    const HeightPair h_pair, const bool& is_global_check) {
    // check for boundaries edges
    for (const auto& contour : ContourGraph::boundary_contour_) {
        if (!ContourGraph::IsEdgeOverlapInHeight(h_pair, HeightPair(contour.first, contour.second)))
            continue;
        if (ContourGraph::IsEdgeCollideSegment(contour, bd_cedge)) {
            return false;
        }
    }
    if (!is_global_check) {
        // check for local range polygons
        const Point3D center_p = Point3D((cedge.start_p.x + cedge.end_p.x) / 2.0f,
            (cedge.start_p.y + cedge.end_p.y) / 2.0f, 0.0f);
        for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
            if (poly_ptr->is_pillar) continue;
            if ((poly_ptr->is_robot_inside !=
                    FARUtil::PointInsideAPoly(poly_ptr->vertices, center_p)) ||
                ContourGraph::IsEdgeCollidePoly(poly_ptr->vertices, cedge)) {
                return false;
            }
        }
        // check for unmatched local contours
        for (const auto& contour : ContourGraph::unmatched_contour_) {
            if (ContourGraph::IsEdgeCollideSegment(contour, cedge)) {
                return false;
            }
        }
        // check for any inactive local contours
        for (const auto& contour : ContourGraph::inactive_contour_) {
            if (ContourGraph::IsEdgeCollideSegment(contour, cedge)) {
                return false;
            }
        }
    } else {
        for (const auto& contour : ContourGraph::global_contour_) {
            if (!ContourGraph::IsEdgeOverlapInHeight(
                    h_pair, HeightPair(contour.first, contour.second)))
                continue;
            if (ContourGraph::IsEdgeCollideSegment(contour, cedge)) {
                return false;
            }
        }
        for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
            if (poly_ptr->is_pillar) continue;
            if (ContourGraph::IsEdgeCollidePoly(poly_ptr->vertices, cedge)) {
                return false;
            }
        }
    }
    return true;
}
// 检查两个导航点是否是一个多边形的，一个多边形可以通过轮廓连接
bool ContourGraph::IsNavNodesConnectFromContour(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    if (node_ptr1->is_odom || node_ptr2->is_odom) return false;
    // 陡坡内点虽然有多边形，但是不是多边形上的点,不能参与判断。这个陡坡边界的连接是要经过角度判断的
    if (node_ptr1->terrain_type == TerrainType::TERRAIN_STEEP ||
        node_ptr2->terrain_type == TerrainType::TERRAIN_STEEP ||
        node_ptr1->terrain_type == TerrainType::TERRAIN_STEEP_BOUND ||
        node_ptr2->terrain_type == TerrainType::TERRAIN_STEEP_BOUND)
        return false;
    const CTNodePtr ctnode1 = node_ptr1->ctnode;
    const CTNodePtr ctnode2 = node_ptr2->ctnode;
    if (ctnode1 == NULL || ctnode2 == NULL || ctnode1 == ctnode2) return false;
    return ContourGraph::IsCTNodesConnectFromContour(ctnode1, ctnode2);
}
// 判断两个导航节点是否通过环境轮廓结构可达
bool ContourGraph::IsCTNodesConnectFromContour(const CTNodePtr& ctnode1, const CTNodePtr& ctnode2) {
    if (ctnode1 == ctnode2 || ctnode1->poly_ptr != ctnode2->poly_ptr) return false;
    // check for boundary collision
    const ConnectPair cedge = ConnectPair(ctnode1->position, ctnode2->position);
    for (const auto& contour : ContourGraph::boundary_contour_) {
        if (ContourGraph::IsEdgeCollideSegment(contour, cedge)) {
            return false;
        }
    }
    // forward search
    CTNodePtr next_ctnode = ctnode1->front;
    while (next_ctnode != NULL && next_ctnode != ctnode1) {
        if (next_ctnode == ctnode2) {
            return true;
        }
        if (next_ctnode->is_global_match ||
            !FARUtil::IsInCylinder(ctnode1->position, ctnode2->position, next_ctnode->position,
                FARUtil::kNearDist, true)) {
            break;
        } else {
            next_ctnode = next_ctnode->front;
        }
    }
    // backward search
    next_ctnode = ctnode1->back;
    while (next_ctnode != NULL && next_ctnode != ctnode1) {
        if (next_ctnode == ctnode2) {
            return true;
        }
        if (next_ctnode->is_global_match ||
            !FARUtil::IsInCylinder(ctnode1->position, ctnode2->position, next_ctnode->position,
                FARUtil::kNearDist, true)) {
            break;
        } else {
            next_ctnode = next_ctnode->back;
        }
    }
    return false;
}
// 找到轮廓中第一个匹配到的ct节点
CTNodePtr ContourGraph::FirstMatchedCTNode(const CTNodePtr& ctnode_ptr) {
    if (ctnode_ptr->is_global_match) return ctnode_ptr;
    CTNodePtr cur_ctnode_ptr = ctnode_ptr->front;
    while (cur_ctnode_ptr != ctnode_ptr) {
        if (cur_ctnode_ptr->is_global_match) return cur_ctnode_ptr;
        cur_ctnode_ptr = cur_ctnode_ptr->front;
    }
    return NULL;
}
// 在一组“近邻导航节点”中，找到一个与“超范围导航节点”通过轮廓结构可以合理连接的近邻节点，并返回这个近邻节点
NavNodePtr ContourGraph::MatchOutrangeNodeWithCTNode(
    const NavNodePtr& out_node_ptr, const NodePtrStack& near_nodes) {
    if (near_nodes.empty()) return NULL;
    float min_dist = FARUtil::kINF;
    NavNodePtr min_matched_node = NULL;
    for (const auto& node_ptr : near_nodes) {
        if (!node_ptr->is_contour_match) continue;
        CTNodePtr matched_ctnode = NULL;
        if (IsContourLineMatch(node_ptr, out_node_ptr, matched_ctnode)) {
            const float dist = FARUtil::VerticalDistToLine2D(
                node_ptr->position, matched_ctnode->position, out_node_ptr->position);
            if (dist < min_dist) {
                min_dist = dist;
                min_matched_node = node_ptr;
            }
        }
    }
    if (min_dist < FARUtil::kNavClearDist) {
        return min_matched_node;
    }
    return NULL;
}

// 判断两个导航节点（一个近邻节点 inNode_ptr，一个超范围节点
// outNode_ptr）之间是否可以通过轮廓结构进行连接，即它们的连线方向是否与环境轮廓的某个段落高度对齐，并且连接线不与障碍物发生碰撞。
// 如果两个节点应该连接，那么它们的连线方向应该与环境轮廓的某个段落高度对齐。
// 近邻导航节点 ←→ 轮廓多边形上的某个点 ←→ 超范围导航节点
bool ContourGraph::IsContourLineMatch(
    const NavNodePtr& inNode_ptr, const NavNodePtr& outNode_ptr, CTNodePtr& matched_ctnode) {
    // 关键思想：如果两个节点应该连接，那么它们的连线方向应该与环境轮廓的某个段落高度对齐。
    // 近邻导航节点 ←→ 轮廓多边形上的某个点 ←→ 超范围导航节点
    // 由于outNode_ptr是超出范围的轮廓点，如果可以连接，那outNode_ptr应该也是轮廓匹配的点，inNode_ptr应该如果是一个轮廓就能连上
    // 轮廓多边形可能很大，跨越多个处理范围
    const CTNodePtr ctnode_ptr = inNode_ptr->ctnode;
    matched_ctnode = NULL;
    // 地形节点跳过轮廓线匹配
    if (ctnode_ptr->terrain_type == TERRAIN_STEEP || ctnode_ptr->terrain_type == TERRAIN_MODERATE ||
        ctnode_ptr->terrain_type == TERRAIN_FLAT ||
        ctnode_ptr->terrain_type == TERRAIN_STEEP_BOUND) {
        return false;
    }
    if (ctnode_ptr == NULL || ctnode_ptr->poly_ptr->is_pillar) return false;
    // check forward
    const PointPair line1(inNode_ptr->position, outNode_ptr->position);
    CTNodePtr next_ctnode = ctnode_ptr->front;
    CTNodePtr prev_ctnode = ctnode_ptr;
    while (!next_ctnode->is_global_match && next_ctnode != ctnode_ptr &&
           FARUtil::IsInCylinder(ctnode_ptr->position, next_ctnode->position, prev_ctnode->position,
               FARUtil::kNearDist, true)) {
        if (!FARUtil::IsPointInMarginRange(next_ctnode->position) &&
            (ctnode_ptr->position - next_ctnode->position).norm_flat() > FARUtil::kMatchDist) {
            const PointPair line2(ctnode_ptr->position, next_ctnode->position);
            if (FARUtil::LineMatchPercentage(line1, line2) > 0.99f) {
                if (IsCTMatchLineFreePolygon(next_ctnode, outNode_ptr, true)) {
                    matched_ctnode = next_ctnode;
                    return true;
                }
            }
        }
        prev_ctnode = next_ctnode;
        next_ctnode = next_ctnode->front;
    }
    // check backward
    next_ctnode = ctnode_ptr->back;
    prev_ctnode = ctnode_ptr;
    while (!next_ctnode->is_global_match && next_ctnode != ctnode_ptr &&
           FARUtil::IsInCylinder(ctnode_ptr->position, next_ctnode->position, prev_ctnode->position,
               FARUtil::kNearDist, true)) {
        if (!FARUtil::IsPointInMarginRange(next_ctnode->position) &&
            (ctnode_ptr->position - next_ctnode->position).norm_flat() > FARUtil::kMatchDist) {
            const PointPair line2(ctnode_ptr->position, next_ctnode->position);
            if (FARUtil::LineMatchPercentage(line1, line2) > 0.99f) {
                if (IsCTMatchLineFreePolygon(next_ctnode, outNode_ptr, true)) {
                    matched_ctnode = next_ctnode;
                    return true;
                }
            }
        }
        prev_ctnode = next_ctnode;
        next_ctnode = next_ctnode->back;
    }
    return false;
}
// 判断在同一个障碍物多边形上，两个已匹配的轮廓节点（ctnode1 和
// ctnode2）之间，沿着轮廓链表（front方向）是否存在“连续的轮廓链路”，即中间没有被阻断。
// 如果两个已匹配的轮廓节点之间链路被阻断，就会把阻断点标记为“必要轮廓节点”，以便后续补充导航点，保证轮廓结构完整。
bool ContourGraph::IsCTNodesConnectWithinOrder(
    const CTNodePtr& ctnode1, const CTNodePtr& ctnode2, CTNodePtr& block_vertex) {
    block_vertex = NULL;
    if (ctnode1 == ctnode2 || ctnode1->poly_ptr != ctnode2->poly_ptr) return false;
    CTNodePtr next_ctnode = ctnode1->front;  // forward search
    while (next_ctnode != NULL && next_ctnode != ctnode2) {
        if (!FARUtil::IsInCylinder(ctnode1->position, ctnode2->position, next_ctnode->position,
                FARUtil::kNearDist, true)) {
            block_vertex = next_ctnode;
            return false;
        }
        next_ctnode = next_ctnode->front;
    }
    return true;
}
// 检查匹配到导航点的ct点的完整性和合理性，防止过渡简化导致轮廓缺失
void ContourGraph::EnclosePolygonsCheck() {
    for (const auto& ctnode_ptr : ContourGraph::polys_ctnodes_) {  // loop each polygon
        if (ctnode_ptr->poly_ptr->is_pillar) continue;
        // 找到该多边形上第一个已匹配到导航点的轮廓节点
        const CTNodePtr start_ctnode_ptr = FirstMatchedCTNode(ctnode_ptr);
        if (start_ctnode_ptr == NULL) continue;
        // 沿着多边形的轮廓节点链表（front指针）循环。
        CTNodePtr pre_ctnode_ptr = start_ctnode_ptr;
        CTNodePtr cur_ctnode_ptr = start_ctnode_ptr->front;
        while (cur_ctnode_ptr != start_ctnode_ptr) {
            if (!cur_ctnode_ptr->is_global_match) {
                cur_ctnode_ptr = cur_ctnode_ptr->front;
                continue;
            }
            // 检查相邻已匹配节点之间的轮廓连接完整性
            CTNodePtr block_vertex = NULL;
            // 检查两个已匹配节点之间的轮廓链是否被阻断
            if (!IsCTNodesConnectWithinOrder(pre_ctnode_ptr, cur_ctnode_ptr, block_vertex) &&
                block_vertex != NULL) {
                // 如果发现阻断点且该点是地面关联且在边缘范围，则标记该点为必要轮廓节点（is_contour_necessary
                // = true），以便后续重新生成导航点。
                if (block_vertex->is_ground_associate &&
                    FARUtil::IsPointInMarginRange(block_vertex->position)) {
                    block_vertex->is_contour_necessary = true;
                }
            }
            pre_ctnode_ptr = cur_ctnode_ptr;
            cur_ctnode_ptr = cur_ctnode_ptr->front;
        }
    }
}
// 创建障碍物的ct点
void ContourGraph::CreateCTNodeObs(
    const Point3D& pos, CTNodePtr& ctnode_ptr, const PolygonPtr& poly_ptr, const bool& is_pillar) {
    ctnode_ptr = std::make_shared<CTNode>();
    ctnode_ptr->position = pos;
    ctnode_ptr->front = NULL;
    ctnode_ptr->back = NULL;
    ctnode_ptr->is_global_match = false;
    ctnode_ptr->is_contour_necessary = false;
    ctnode_ptr->is_ground_associate = false;
    ctnode_ptr->nav_node_id = 0;
    ctnode_ptr->poly_ptr = poly_ptr;
    ctnode_ptr->free_direct = is_pillar ? NodeFreeDirect::PILLAR : NodeFreeDirect::UNKNOW;
    ctnode_ptr->connect_nodes.clear();
    // 此节点是障碍物
    ctnode_ptr->terrain_type = TerrainType::TERRAIN_OBSTACLE;
}
// 创建其他类型的ct点
void ContourGraph::CreatCTNodeTerrain(
    const Point3D& pos, CTNodePtr& ctnode_ptr, const bool& is_pillar, TerrainType terrain_type) {
    ctnode_ptr = std::make_shared<CTNode>();
    ctnode_ptr->position = pos;
    ctnode_ptr->front = NULL;
    ctnode_ptr->back = NULL;
    ctnode_ptr->is_global_match = false;
    ctnode_ptr->is_contour_necessary = true;
    ctnode_ptr->is_ground_associate = false;
    ctnode_ptr->nav_node_id = 0;
    ctnode_ptr->free_direct = is_pillar ? NodeFreeDirect::PILLAR : NodeFreeDirect::UNKNOW;
    ctnode_ptr->connect_nodes.clear();
    // 此节点是特殊地形
    ctnode_ptr->terrain_type = terrain_type;
    ctnode_ptr->is_terrain = true;

    MapHandler::GetGradientAtPosition(
        ctnode_ptr->position, ctnode_ptr->gradient.x, ctnode_ptr->gradient.y, ctnode_ptr->slop);
}
// 创建多边形
void ContourGraph::CreatePolygon(const PointStack& poly_points, PolygonPtr& poly_ptr) {
    poly_ptr = std::make_shared<Polygon>();
    poly_ptr->N = poly_points.size();
    poly_ptr->vertices = poly_points;
    poly_ptr->is_robot_inside = FARUtil::PointInsideAPoly(poly_points, odom_node_ptr_->position);
    float perimeter = 0.0f;
    poly_ptr->is_pillar = this->IsAPillarPolygon(poly_points, perimeter);
    poly_ptr->perimeter = perimeter;
}
// 寻找距离最近的导航点
NavNodePtr ContourGraph::NearestNavNodeForCTNode(
    const CTNodePtr& ctnode_ptr, const NodePtrStack& near_nodes) {
    float nearest_dist = FARUtil::kINF;
    NavNodePtr nearest_node = NULL;
    float min_edist = FARUtil::kINF;
    const float dir_thred = 0.5f;  // cos(pi/3);
    for (const auto& node_ptr : near_nodes) {
        if (node_ptr->is_odom || node_ptr->is_navpoint || FARUtil::IsOutsideGoal(node_ptr) ||
            !IsInMatchHeight(ctnode_ptr, node_ptr))
            continue;
        // [新增] 地形类型必须匹配
        if (ctnode_ptr->terrain_type != node_ptr->terrain_type) {
            continue;
        }

        // no match with pillar to non-pillar local vertices
        if ((node_ptr->free_direct == NodeFreeDirect::PILLAR &&
                ctnode_ptr->free_direct != NodeFreeDirect::PILLAR) ||
            (ctnode_ptr->free_direct == NodeFreeDirect::PILLAR &&
                node_ptr->free_direct != NodeFreeDirect::PILLAR)) {
            continue;
        }
        float dist_thred = FARUtil::kMatchDist;
        float dir_score = 0.0f;
        if (ctnode_ptr->free_direct != NodeFreeDirect::PILLAR &&
            node_ptr->free_direct != NodeFreeDirect::UNKNOW &&
            node_ptr->free_direct != NodeFreeDirect::PILLAR) {
            if (ctnode_ptr->free_direct == node_ptr->free_direct) {
                const Point3D topo_dir1 = FARUtil::SurfTopoDirect(node_ptr->surf_dirs);
                const Point3D topo_dir2 = FARUtil::SurfTopoDirect(ctnode_ptr->surf_dirs);
                dir_score = (topo_dir1 * topo_dir2 - dir_thred) / (1.0f - dir_thred);
            }
        } else if (node_ptr->free_direct == NodeFreeDirect::PILLAR &&
                   ctnode_ptr->free_direct == NodeFreeDirect::PILLAR) {
            dir_score = 0.5f;
        }
        dist_thred *= dir_score;
        const float edist = (node_ptr->position - ctnode_ptr->position).norm_flat();
        if (edist < dist_thred && edist < min_edist) {
            nearest_node = node_ptr;
            min_edist = edist;
        }
    }
    if (nearest_node != NULL && nearest_node->is_contour_match) {
        const float pre_dist =
            (nearest_node->position - nearest_node->ctnode->position).norm_flat();
        if (min_edist < pre_dist) {
            // reset matching for previous ctnode
            RemoveMatchWithNavNode(nearest_node);
        } else {
            nearest_node = NULL;
        }
    }
    return nearest_node;
}
// 分析ct点的表面方向和凹凸性
void ContourGraph::AnalysisSurfAngleAndConvexity(const CTNodeStack& contour_graph) {
    for (const auto& ctnode_ptr : contour_graph) {
        if (ctnode_ptr->free_direct == NodeFreeDirect::PILLAR || ctnode_ptr->poly_ptr->is_pillar) {
            ctnode_ptr->surf_dirs = {Point3D(0, 0, -1), Point3D(0, 0, -1)};
            ctnode_ptr->poly_ptr->is_pillar = true;
            ctnode_ptr->free_direct = NodeFreeDirect::PILLAR;
        } else {
            CTNodePtr next_ctnode;
            // front direction
            next_ctnode = ctnode_ptr->front;
            Point3D start_p = ctnode_ptr->position;
            Point3D end_p = next_ctnode->position;
            float edist = (end_p - ctnode_ptr->position).norm_flat();
            while (next_ctnode != NULL && next_ctnode != ctnode_ptr &&
                   edist < FARUtil::kNavClearDist) {
                next_ctnode = next_ctnode->front;
                start_p = end_p;
                end_p = next_ctnode->position;
                edist = (end_p - ctnode_ptr->position).norm_flat();
            }
            if (edist < FARUtil::kNavClearDist) {  // This Node should be a pillar.
                ctnode_ptr->surf_dirs = {Point3D(0, 0, -1), Point3D(0, 0, -1)};
                ctnode_ptr->poly_ptr->is_pillar = true;
                ctnode_ptr->free_direct = NodeFreeDirect::PILLAR;
                continue;
            } else {
                ctnode_ptr->surf_dirs.first = FARUtil::ContourSurfDirs(
                    end_p, start_p, ctnode_ptr->position, FARUtil::kNavClearDist);
            }
            // back direction
            next_ctnode = ctnode_ptr->back;
            start_p = ctnode_ptr->position;
            end_p = next_ctnode->position;
            edist = (end_p - ctnode_ptr->position).norm_flat();
            while (next_ctnode != NULL && next_ctnode != ctnode_ptr &&
                   edist < FARUtil::kNavClearDist) {
                next_ctnode = next_ctnode->back;
                start_p = end_p;
                end_p = next_ctnode->position;
                edist = (end_p - ctnode_ptr->position).norm_flat();
            }
            if (edist < FARUtil::kNavClearDist) {  // This Node should be a pillar.
                ctnode_ptr->surf_dirs = {Point3D(0, 0, -1), Point3D(0, 0, -1)};  // TODO!
                ctnode_ptr->poly_ptr->is_pillar = true;
                ctnode_ptr->free_direct = NodeFreeDirect::PILLAR;
                continue;
            } else {
                ctnode_ptr->surf_dirs.second = FARUtil::ContourSurfDirs(
                    end_p, start_p, ctnode_ptr->position, FARUtil::kNavClearDist);
            }
        }
        // analysis convexity (except pillar)
        this->AnalysisConvexityOfCTNode(ctnode_ptr);
    }
}
// 根据周长判断是不是pillar
bool ContourGraph::IsAPillarPolygon(const PointStack& vertex_points, float& perimeter) {
    perimeter = 0.0f;
    if (vertex_points.size() < 3) return true;
    Point3D prev_p(vertex_points[0]);
    for (std::size_t i = 1; i < vertex_points.size(); i++) {
        const Point3D cur_p(vertex_points[i]);
        const float dist = std::hypotf(cur_p.x - prev_p.x, cur_p.y - prev_p.y);
        perimeter += dist;
        prev_p = cur_p;
    }
    return perimeter > ctgraph_params_.kPillarPerimeter ? false : true;
}
// 两条线段是否相交  true:相交   false:不相交
bool ContourGraph::IsEdgeCollideSegment(const PointPair& line, const ConnectPair& edge) {
    const cv::Point2f start_p(line.first.x, line.first.y);
    const cv::Point2f end_p(line.second.x, line.second.y);
    if (POLYOPS::doIntersect(start_p, end_p, edge.start_p, edge.end_p)) {
        return true;
    }
    return false;
}
// 是否和障碍物多边形相交  true:相交  false:不相交
bool ContourGraph::IsEdgeCollidePoly(const PointStack& poly, const ConnectPair& edge) {
    const int N = poly.size();
    if (N < 3) cout << "Poly vertex size less than 3." << endl;
    for (int i = 0; i < N; i++) {
        const PointPair line(poly[i], poly[FARUtil::Mod(i + 1, N)]);
        if (ContourGraph::IsEdgeCollideSegment(line, edge)) {
            return true;
        }
    }
    return false;
}
// 分析点的凹凸性
void ContourGraph::AnalysisConvexityOfCTNode(const CTNodePtr& ctnode_ptr) {
    if (ctnode_ptr->surf_dirs.first == Point3D(0, 0, -1) ||
        ctnode_ptr->surf_dirs.second == Point3D(0, 0, -1) || ctnode_ptr->poly_ptr->is_pillar) {
        ctnode_ptr->surf_dirs.first = Point3D(0, 0, -1),
        ctnode_ptr->surf_dirs.second == Point3D(0, 0, -1);
        ctnode_ptr->poly_ptr->is_pillar = true;
        ctnode_ptr->free_direct = NodeFreeDirect::PILLAR;
        return;
    }
    bool is_wall = false;
    const Point3D topo_dir = FARUtil::SurfTopoDirect(ctnode_ptr->surf_dirs, is_wall);
    if (is_wall) {
        ctnode_ptr->free_direct = NodeFreeDirect::UNKNOW;
        return;
    }
    const Point3D ev_p = ctnode_ptr->position + topo_dir * FARUtil::kLeafSize;
    if (FARUtil::IsConvexPoint(ctnode_ptr->poly_ptr, ev_p)) {
        ctnode_ptr->free_direct = NodeFreeDirect::CONVEX;
    } else {
        ctnode_ptr->free_direct = NodeFreeDirect::CONCAVE;
    }
}
// 如果一个点在障碍物多边形（非pillar）内部，并且机器人当前不在该多边形内，则把这个点“投影”到多边形外部的自由空间，以保证点不落在障碍物内部。
// 防止路径点、采样点、导航点落在障碍物内部
// 用于把落在障碍物多边形内部的点，自动“推”到多边形外部的自由空间，防止路径点或采样点非法穿入障碍物。
bool ContourGraph::ReprojectPointOutsidePolygons(Point3D& point, const float& free_radius) {
    PolygonPtr inside_poly_ptr = NULL;
    bool is_inside_poly = false;
    // 遍历多边形
    for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
        if (poly_ptr->is_pillar) continue;
        // 如果点在多边形内但机器人位置不再多边形内
        if (FARUtil::PointInsideAPoly(poly_ptr->vertices, point) &&
            !FARUtil::PointInsideAPoly(poly_ptr->vertices, FARUtil::free_odom_p)) {
            inside_poly_ptr = poly_ptr;
            is_inside_poly = true;
            break;
        }
    }
    // 如果在内部，寻找最近的凸顶点
    if (is_inside_poly) {
        float near_dist = FARUtil::kINF;
        Point3D reproject_p = point;
        Point3D free_dir(0, 0, -1);
        const int N = inside_poly_ptr->vertices.size();
        for (int idx = 0; idx < N; idx++) {
            const Point3D vertex = inside_poly_ptr->vertices[idx];
            const float temp_dist = (vertex - point).norm_flat();
            if (temp_dist < near_dist) {
                const Point3D dir1 =
                    (inside_poly_ptr->vertices[FARUtil::Mod(idx - 1, N)] - vertex).normalize_flat();
                const Point3D dir2 =
                    (inside_poly_ptr->vertices[FARUtil::Mod(idx + 1, N)] - vertex).normalize_flat();
                const Point3D dir = (dir1 + dir2).normalize_flat();
                if (FARUtil::PointInsideAPoly(
                        inside_poly_ptr->vertices, vertex + dir * FARUtil::kLeafSize)) {  // convex
                    reproject_p = vertex;
                    near_dist = temp_dist;
                    free_dir = dir;
                }
            }
        }
        const float origin_z = point.z;
        point = reproject_p - free_dir * free_radius;
        point.z = origin_z;
    }
    return is_inside_poly;
}

void ContourGraph::AddContourToSets(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    NavEdge edge(node_ptr1, node_ptr2);
    // force to form pair id1 < id2
    if (node_ptr1->id > node_ptr2->id) edge = NavEdge(node_ptr2, node_ptr1);

    ContourGraph::global_contour_set_.insert(edge);
    if (node_ptr1->is_boundary && node_ptr2->is_boundary) {
        ContourGraph::boundary_contour_set_.insert(edge);
    }
}

void ContourGraph::DeleteContourFromSets(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2) {
    NavEdge edge(node_ptr1, node_ptr2);
    // force to form pair id1 < id2
    if (node_ptr1->id > node_ptr2->id) edge = NavEdge(node_ptr2, node_ptr1);

    ContourGraph::global_contour_set_.erase(edge);
    if (node_ptr1->is_boundary && node_ptr2->is_boundary) {
        ContourGraph::boundary_contour_set_.erase(edge);
    }
}

void ContourGraph::ExtractGlobalContours() {
    ContourGraph::global_contour_.clear();
    ContourGraph::inactive_contour_.clear();
    ContourGraph::unmatched_contour_.clear();
    ContourGraph::boundary_contour_.clear();
    ContourGraph::local_boundary_.clear();
    for (const auto& edge : ContourGraph::global_contour_set_) {
        ContourGraph::global_contour_.push_back({edge.first->position, edge.second->position});
        if (IsEdgeInLocalRange(edge.first, edge.second)) {
            if (!this->IsActiveEdge(edge.first, edge.second)) {
                ContourGraph::inactive_contour_.push_back(
                    {edge.first->position, edge.second->position});
            } else if (!edge.first->is_near_nodes || !edge.second->is_near_nodes) {
                PointPair unmatched_pair =
                    std::make_pair(edge.first->position, edge.second->position);
                if (edge.first->is_contour_match) {
                    unmatched_pair.first = edge.first->ctnode->position;
                } else if (edge.second->is_contour_match) {
                    unmatched_pair.second = edge.second->ctnode->position;
                }
                ContourGraph::unmatched_contour_.push_back(unmatched_pair);
            }
        }
    }
    for (const auto& edge : ContourGraph::boundary_contour_set_) {
        ContourGraph::boundary_contour_.push_back({edge.first->position, edge.second->position});
        if (IsEdgeInLocalRange(edge.first, edge.second)) {
            ContourGraph::local_boundary_.push_back({edge.first->position, edge.second->position});
            bool is_new_invalid = false;
            if (!IsValidBoundary(edge.first, edge.second, is_new_invalid) && is_new_invalid) {
                edge.first->invalid_boundary.insert(edge.second->id);
                edge.second->invalid_boundary.insert(edge.first->id);
            }
        }
    }
}

bool ContourGraph::IsValidBoundary(
    const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2, bool& is_new) {
    is_new = true;
    if (node_ptr1->invalid_boundary.find(node_ptr2->id) !=
        node_ptr1->invalid_boundary.end()) {  // already invalid
        is_new = false;
        return false;
    }
    // check against local polygon
    const ConnectPair cedge = ConnectPair(node_ptr1->position, node_ptr2->position);
    for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
        if (poly_ptr->is_pillar) continue;
        if (ContourGraph::IsEdgeCollidePoly(poly_ptr->vertices, cedge)) {
            return false;
        }
    }
    return true;
}

void ContourGraph::UpdateOdomFreePosition(const NavNodePtr& odom_ptr, Point3D& global_free_p) {
    Point3D free_p = odom_ptr->position;
    bool is_free_p = true;
    PointStack free_sample_points;
    for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
        if (!poly_ptr->is_pillar && poly_ptr->is_robot_inside) {
            is_free_p = false;
            FARUtil::CreatePointsAroundCenter(
                free_p, FARUtil::kNavClearDist, FARUtil::kLeafSize, free_sample_points);
            break;
        }
    }
    if (is_free_p) is_robot_inside_poly_ = false;
    global_free_p = free_p;
    if (!is_free_p && !is_robot_inside_poly_) {
        bool is_free_pos_found = false;
        for (const auto& p : free_sample_points) {
            bool is_sample_free = true;
            for (const auto& poly_ptr : ContourGraph::contour_polygons_) {
                if (!poly_ptr->is_pillar && FARUtil::PointInsideAPoly(poly_ptr->vertices, p)) {
                    is_sample_free = false;
                    break;
                }
            }
            if (is_sample_free) {
                global_free_p = p;
                is_free_pos_found = true;
                break;
            }
        }
        if (!is_free_pos_found) is_robot_inside_poly_ = true;
    }
}

ConnectPair ContourGraph::ReprojectEdge(
    const CTNodePtr& ctnode_ptr1, const NavNodePtr& node_ptr2, const float& dist) {
    ConnectPair edgeOut;
    const float ndist = (ctnode_ptr1->position - node_ptr2->position).norm_flat();
    const float ref_dist = std::min(ndist * 0.4f, dist);

    edgeOut.start_p = ProjectNode(ctnode_ptr1, ref_dist);  // node 1
    edgeOut.end_p = ProjectNode(node_ptr2, ref_dist);      // node 2

    return edgeOut;
}

ConnectPair ContourGraph::ReprojectEdge(const NavNodePtr& node_ptr1, const NavNodePtr& node_ptr2,
    const float& dist, const bool& is_global_check) {
    ConnectPair edgeOut;
    const float ndist = (node_ptr1->position - node_ptr2->position).norm_flat();
    const float ref_dist = std::min(ndist * 0.4f, dist);
    // node 1
    if (!is_global_check && node_ptr1->is_contour_match &&
        node_ptr1->ctnode->free_direct == node_ptr1->free_direct) {  // node 1
        const auto ctnode1 = node_ptr1->ctnode;
        edgeOut.start_p = ProjectNode(ctnode1, ref_dist);  // node 1
    } else {
        edgeOut.start_p = ProjectNode(node_ptr1, ref_dist);  // node 1
    }
    // node 2
    if (!is_global_check && node_ptr2->is_contour_match &&
        node_ptr2->ctnode->free_direct == node_ptr2->free_direct) {  // node 2
        const auto ctnode2 = node_ptr2->ctnode;
        edgeOut.end_p = ProjectNode(ctnode2, ref_dist);  // node 1
    } else {
        edgeOut.end_p = ProjectNode(node_ptr2, ref_dist);
    }
    return edgeOut;
}

void ContourGraph::ResetCurrentContour() {
    this->ClearContourGraph();
    // clear contour sets
    ContourGraph::global_contour_set_.clear();
    ContourGraph::boundary_contour_set_.clear();

    odom_node_ptr_ = NULL;
    is_robot_inside_poly_ = false;
}
