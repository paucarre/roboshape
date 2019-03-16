#pragma once

#include "datatypes.hpp"

namespace roboshape { 

    void extract_point_index_to_neighbours(pcl::PolygonMesh &polygon_mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud,    
            std::shared_ptr<std::map<uint32_t, std::vector<int>>> point_index_to_neighbours) {
        cout << "Extracting point to neighbours map..." << std::endl;
        Mesh::VertexIndices vertex_indices;            
        Mesh mesh;
        roboshape::generate_mesh(polygon_mesh, point_cloud, mesh, vertex_indices);
        for(uint32_t point_index = 0 ; point_index < vertex_indices.size() ; ++point_index) {
            auto outgoing_half_edges = mesh.getOutgoingHalfEdgeAroundVertexCirculator(vertex_indices[point_index]);
            auto outgoing_half_edges_end = outgoing_half_edges;
            do {
                Mesh::HalfEdgeIndex target_edge = outgoing_half_edges.getTargetIndex();
                auto neighbour_point_index = mesh.getTerminatingVertexIndex(target_edge).get();
                auto vertices_iterator = point_index_to_neighbours->find(point_index);
                if (vertices_iterator == point_index_to_neighbours->end()) {
                    (*point_index_to_neighbours)[point_index] = { neighbour_point_index };
                } else {
                    vertices_iterator->second.push_back(neighbour_point_index);
                }
            } while (++outgoing_half_edges != outgoing_half_edges_end);
        }
        cout << "... point to neighbours map extraction finished" << std::endl;
    }

    boost::optional<float> get_radius_of_curvature(Eigen::Vector3f origin_point, Eigen::Vector3f origin_normal, pcl::PointXYZRGBNormal neighbour) {
        Eigen::Vector3f neighbour_point(neighbour.x, neighbour.y, neighbour.z);
        Eigen::Vector3f neighbour_normal(neighbour.normal_x, neighbour.normal_y, neighbour.normal_z);
        // radius
        auto origin_to_neighbour_dist = (origin_point - neighbour_point).norm();
        if(origin_to_neighbour_dist > 1e-3) {
            float normal_distance = (neighbour_normal  - origin_normal).norm();
            float radius_of_curvature =   normal_distance / origin_to_neighbour_dist;
            return radius_of_curvature;
        }
        return {};
    }

    boost::optional<float> extract_curvature(roboshape::Primitive primitive, int point_index, std::vector<int> neighbour_indices, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals_ptr) {
        /**
         * If it has 6 neighbours the orthogonal basis of the tangent space is the cross on U-V ignoring the two diagonal connections
         * 
         *    *   O   
         *      \ |  
         *    O---P---O
         *        | \
         *        O   *
         * 
         * 
         * If it has 4 neighbours the orthogonal basis consists of all the connections skipping the diagonal ones
         * 
         *                 O---P---O 
         *                     | \
         *                     O   *
         *  
         *     O                           *   O
         *     |                             \ |
         *     P---O                       O---P
         *     | \                             | 
         *     O   *                           O
         * 
         *                *   O   
         *                  \ | 
         *                O---P---O 
         * 
         * 
         * If it has 3 neighbours the orthogonal basis consists of all the connections skipping the diagonal ones
         * 
         *     P---O
         *     | \
         *     O  * 
         *  
         *                  *   O  
         *                    \ |
         *                  O---P
         * 
         * 
         * 
         * If it has 2 neighbours the orthogonal basis consists of all the connections 
         * 
         *                  O---P
         *                      |
         *                      O 
         *  
         *     O    
         *     | 
         *     P---O
         * 
         */ 

        pcl::PointCloud<pcl::PointXYZRGBNormal> &normals = *normals_ptr;    
        // normal-points
        pcl::PointXYZRGBNormal center  = normals[point_index];
        std::vector<float> curvatures;
        Eigen::Vector3f center_point(center.x, center.y, center.z);
        Eigen::Vector3f center_normal(center.normal_x, center.normal_y, center.normal_z);
        auto to_point_normal = [&normals] (int index) { return normals[index]; };
        auto to_radius_of_curvature = [&center_point, &center_normal] (pcl::PointXYZRGBNormal neighbour) {
                return get_radius_of_curvature(center_point, center_normal, neighbour);
        };
        auto insert_into_curvature = [&curvatures](float curvature) { 
                curvatures.push_back(curvature);
                return true;
        };
        primitive.index_down(point_index).map(to_point_normal).flat_map(to_radius_of_curvature).map(insert_into_curvature);
        primitive.index_up(point_index).map(to_point_normal).flat_map(to_radius_of_curvature).map(insert_into_curvature);
        primitive.index_left(point_index).map(to_point_normal).flat_map(to_radius_of_curvature).map(insert_into_curvature);
        primitive.index_right(point_index).map(to_point_normal).flat_map(to_radius_of_curvature).map(insert_into_curvature);
        if(curvatures.size() > 0) {
            float sum_curvatures = 0.0;
            for(auto curvature : curvatures) {
                sum_curvatures = sum_curvatures + curvature;
            }
            auto mean_curvature = sum_curvatures / (float)curvatures.size();
            return mean_curvature; 
        } else {
            return {};
        }
    }

    void extract_planes(std::vector<roboshape::Primitive> &primitives, 
            pcl::PolygonMesh &mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud, 
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals)  {
        std::shared_ptr<std::map<uint32_t, std::vector<int>>> point_index_to_neighbours (new std::map<uint32_t, std::vector<int>>);
        cout << "Starting plane extraction..." << std::endl; 
        extract_point_index_to_neighbours(mesh, point_cloud, point_index_to_neighbours);
        for(auto primitive : primitives) {
            for(uint32_t point_index = primitive.first ; point_index <= primitive.last ; ++point_index) {
                auto neighbours = point_index_to_neighbours->find(point_index);
                if(neighbours != point_index_to_neighbours->end()) {
                auto point_and_neighbours = *neighbours;
                    auto curvature_opt = extract_curvature(primitive, point_and_neighbours.first, point_and_neighbours.second, normals);
                    if(curvature_opt) { 
                        float curvature = *curvature_opt;
                        if(curvature > 1.0) {
                            uint8_t r = 200, g = 50, b = 50;
                            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                            (   *normals)[point_and_neighbours.first].rgb = *reinterpret_cast<float*>(&rgb);
                        } else {
                            uint8_t r = 0, g = 0, b = 255;
                            uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                            (   *normals)[point_and_neighbours.first].rgb = *reinterpret_cast<float*>(&rgb);
                        }
                    }
                }
            }
        }
        cout << "... end of plane extraction." << std::endl;     
    }

}
