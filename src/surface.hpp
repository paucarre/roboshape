#pragma once

#include "datatypes.hpp"

namespace roboshape { 

    void invert_normals(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals, 
        std::vector<roboshape::Primitive> &primitive_borders,  uint32_t primitive_border_index) {
        auto current_primitive_borders = primitive_borders[primitive_border_index];
        for(uint32_t vertex_index = current_primitive_borders.first ; vertex_index <= current_primitive_borders.last ; ++vertex_index) {
            pcl::PointXYZRGBNormal& point_normal = (*output_normals)[vertex_index];
            point_normal.normal_x = - point_normal.normal_x;
            point_normal.normal_y = - point_normal.normal_y;
            point_normal.normal_z = - point_normal.normal_z;
        }
    }

    boost::optional<uint32_t> viewpoint_normal_consistency(pcl::PointXYZ viewpoint,
            pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
            std::vector<roboshape::Primitive> &primitive_borders,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals) {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(point_cloud);
        int K = 5;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        if ( kdtree.nearestKSearch(viewpoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            size_t closest_to_viewpoint = pointIdxNKNSearch[0];
            auto border_index_opt = roboshape::get_primitive_border_index(closest_to_viewpoint, primitive_borders);
            if(border_index_opt) {
                uint32_t border_index = *border_index_opt;
                cout << "Primitive index of point closest to viewpoint: " << border_index << std::endl;
                pcl::PointXYZRGBNormal& closest_to_viewpoint_pointnormal = (*output_normals)[closest_to_viewpoint];
                Eigen::Vector3f closest_to_viewpoint_normal_eigen(closest_to_viewpoint_pointnormal.normal_x, closest_to_viewpoint_pointnormal.normal_y, closest_to_viewpoint_pointnormal.normal_z);
                Eigen::Vector3f closest_to_viewpoint_eigen(closest_to_viewpoint_pointnormal.x, closest_to_viewpoint_pointnormal.y, closest_to_viewpoint_pointnormal.z);
                Eigen::Vector3f viewpoint_eigen(viewpoint.x, viewpoint.y, viewpoint.z);
                Eigen::Vector3f  desired_directional_normal = viewpoint_eigen - closest_to_viewpoint_eigen;
                if (desired_directional_normal.adjoint() * closest_to_viewpoint_normal_eigen < 0.0) {
                    // negative inner product and thus wrong direction, need to invert all normals in primitive
                    invert_normals(output_normals, primitive_borders, border_index);
                }            
            }
            return border_index_opt;
        }
        return {};
    }

    void detect_bordering_points(
                pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, 
                std::vector<roboshape::Primitive> &primitive_borders,
                std::map<uint32_t, std::vector<roboshape::BorderingPoint>> &bordering_points) {
        cout << "Begin detection of bordering points..." << std::endl;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        std::set<uint32_t> processed_points;
        kdtree.setInputCloud(point_cloud);
        pcl::PointXYZ searchPoint;
        int K = 5;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        for(uint32_t primitive_index = 0 ; primitive_index < primitive_borders.size() ; ++primitive_index) {
            std::set<uint32_t> borders = primitive_borders[primitive_index].borders;
            // for each border get the closest node such that it's not in the border
            for(uint32_t const& border_point : borders) {
                if(processed_points.find(border_point) != processed_points.end()) {    
                    pcl::PointXYZ point_in_border = (*point_cloud)[border_point];
                    auto total_found = kdtree.nearestKSearch(point_in_border, K, pointIdxNKNSearch, pointNKNSquaredDistance);
                    if(total_found  > 0) {
                        bool found = false;
                        for(uint32_t current_neighbour = 0 ; !found && current_neighbour < total_found ; current_neighbour++) {                
                            if(processed_points.find(current_neighbour) != processed_points.end()) {    
                                auto current_index = pointIdxNKNSearch[current_neighbour];
                                auto primitive_of_index_opt = roboshape::get_primitive_border_index(current_index, primitive_borders);                    
                                if(primitive_of_index_opt){                        
                                    found = *primitive_of_index_opt != primitive_index;
                                    if(found){
                                        auto current_bordering_point = roboshape::BorderingPoint(border_point, primitive_index, current_index, *primitive_of_index_opt);
                                        if(bordering_points.find(primitive_index) == bordering_points.end()){
                                            bordering_points[primitive_index] = std::vector<roboshape::BorderingPoint>();
                                        }
                                        bordering_points[primitive_index].push_back(current_bordering_point);
                                    }
                                }
                                processed_points.insert(current_neighbour);
                            }
                        }
                    }
                }
                processed_points.insert(border_point);
            }
        }
        cout << "... end of detection of bordering points" << std::endl;
    }

    void make_smooth_manifold(pcl::PolygonMesh &polygon_mesh, 
                pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, 
                std::vector<roboshape::Primitive> &primitive_borders, 
                pcl::PointXYZ viewpoint,
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals) {
        std::map<uint32_t, std::vector<roboshape::BorderingPoint>> bordering_points;
        detect_bordering_points(point_cloud, primitive_borders, bordering_points);
        cout << "Bordering points: " << bordering_points.size() << std::endl;
        roboshape::Mesh::VertexIndices vertex_indices;            
        roboshape::Mesh mesh;
        generate_mesh(polygon_mesh, *point_cloud, mesh, vertex_indices);
        std::set<uint32_t> primiteves_processed;
        std::stack<uint32_t> primitives_to_process;
        boost::optional<uint32_t> border_index_opt = viewpoint_normal_consistency(viewpoint,
            point_cloud, primitive_borders, output_normals);
        if(border_index_opt) {
            uint32_t border_index = *border_index_opt;
            primitives_to_process.push(border_index);
            while(!primitives_to_process.empty()) {
                auto current_border_index = primitives_to_process.top();
                //cout << "Processing primitive: " << current_border_index << std::endl;
                primiteves_processed.insert(current_border_index);
                primitives_to_process.pop();
                auto bordering_points_interator = bordering_points.find(current_border_index);
                if(bordering_points_interator != bordering_points.end()) {
                    std::vector<roboshape::BorderingPoint> bordering_points = (*bordering_points_interator).second;
                    //cout << "Total number of borders: " <<  bordering_points.size() << std::endl;
                    for(roboshape::BorderingPoint bordering_point : bordering_points) {
                        if(primiteves_processed.find(bordering_point.destination_primitive) == primiteves_processed.end()){
                            // get normal from source primitive
                            auto source_normal = (*output_normals)[bordering_point.source_point];
                            auto destination_normal = (*output_normals)[bordering_point.destination_point];
                            Eigen::Vector3f source_normal_eig(source_normal.normal_x, source_normal.normal_y, source_normal.normal_z);
                            Eigen::Vector3f destination_normal_eig(destination_normal.normal_x, destination_normal.normal_y, destination_normal.normal_z);
                            if(source_normal_eig.adjoint() * destination_normal_eig < 0.0) {
                                invert_normals(output_normals, primitive_borders, bordering_point.destination_primitive);
                            }
                            primitives_to_process.push(bordering_point.destination_primitive);
                        }
                    }
                }
                
            }
        }
    }

    pcl::PolygonMesh::Ptr laplacian_smoothing(pcl::PolygonMesh::Ptr mesh) {
        cout << "Starting Laplacian Smoothing..." << std::endl; 
        pcl::PolygonMesh::Ptr smoothed_mesh (new pcl::PolygonMesh());
        pcl::MeshSmoothingLaplacianVTK vtk; 
        vtk.setInputMesh(mesh); 
        vtk.setNumIter(20000); 
        vtk.setConvergence(0.0001); 
        vtk.setRelaxationFactor(0.0001); 
        vtk.setFeatureEdgeSmoothing(true); 
        vtk.setFeatureAngle(M_PI / 5); 
        vtk.setBoundarySmoothing(true); 
        vtk.process(*smoothed_mesh); 
        cout << "... end of Laplacian Smoothing." << std::endl; 
        return smoothed_mesh;
    }

    void extract_point_index_to_normals(pcl::PolygonMesh &polygon_mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud,    
            std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals) {
        roboshape::Mesh::VertexIndices vertex_indices;            
        roboshape::Mesh mesh;
        roboshape::generate_mesh(polygon_mesh, point_cloud, mesh, vertex_indices);
        for(uint32_t point_index = 0 ; point_index < vertex_indices.size() ; ++point_index) {
            auto outgoing_half_edges = mesh.getOutgoingHalfEdgeAroundVertexCirculator(vertex_indices[point_index]);
            auto outgoing_half_edges_end = outgoing_half_edges;
            boost::optional<Eigen::Vector3f> previous_vector = {};
            do {
                roboshape::Mesh::HalfEdgeIndex target_edge = outgoing_half_edges.getTargetIndex();
                pcl::PointXYZ origin = mesh.getVertexDataCloud()[mesh.getOriginatingVertexIndex(target_edge).get()];
                pcl::PointXYZ terminating = mesh.getVertexDataCloud()[mesh.getTerminatingVertexIndex(target_edge).get()];
                Eigen::Vector3f origin_vector(origin.x, origin.y, origin.z);
                Eigen::Vector3f terminating_vector(terminating.x, terminating.y, terminating.z);
                Eigen::Vector3f current_vector = terminating_vector - origin_vector;
                if(previous_vector) {
                    Eigen::Vector3f normal = (*previous_vector).cross(current_vector); 
                    normal.normalize();
                    auto vertices_iterator = point_index_to_normals->find(point_index);
                    if (vertices_iterator == point_index_to_normals->end()) {
                        (*point_index_to_normals)[point_index] = { normal };
                    } else {
                        vertices_iterator->second.push_back(normal);
                    }
                }
                previous_vector = current_vector;
            } while (++outgoing_half_edges != outgoing_half_edges_end);
        }
    }

    void extract_normals(pcl::PolygonMesh &mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud, 
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals)  {
        cout << "Starting normal extraction..." << std::endl; 
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals (new std::map<uint32_t, std::vector<Eigen::Vector3f>>);
        extract_point_index_to_normals(mesh, point_cloud, point_index_to_normals);
        for(uint32_t point_index = 0 ; point_index < mesh.cloud.data.size() ; ++point_index) {
            auto normals_iterator = point_index_to_normals->find(point_index);
            if(normals_iterator != point_index_to_normals->end()) {
                Eigen::Vector3f sum_of_normals = Eigen::Vector3f::Zero();
                for(auto const& normal : normals_iterator->second){
                    sum_of_normals = sum_of_normals + normal;
                }
                sum_of_normals.normalize();
                pcl::PointXYZRGBNormal point_normal = pcl::PointXYZRGBNormal();
                uint8_t r = 0, g = 0, b = 100;
                uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                point_normal.rgb = *reinterpret_cast<float*>(&rgb);
                point_normal.normal_x = sum_of_normals[0];
                point_normal.normal_y = sum_of_normals[1];
                point_normal.normal_z = sum_of_normals[2];
                pcl::PointXYZ point = point_cloud.points[point_index];
                point_normal.x = point.x;
                point_normal.y = point.y;
                point_normal.z = point.z;
                output_normals->points.push_back(point_normal);            
            }
        }
        cout << "... end of normal extraction." << std::endl;     
    }

}