// Point cloud library
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/conversions.h>
#include <pcl/surface/poisson.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/processing.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/geometry/triangle_mesh.h>
#include <pcl/geometry/quad_mesh.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/geometry/mesh_conversion.h>
#include <pcl/point_cloud.h>


// Boost library
#include <boost/optional.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/config.hpp>
#include <boost/program_options/environment_iterator.hpp>
#include <boost/program_options/eof_iterator.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/program_options/option.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/version.hpp>
#include <boost/thread/thread.hpp>
#include <boost/optional/optional_io.hpp>

#include <numeric>  
#include <iostream>

struct BorderingPoint {
  
  uint32_t source_point;
  uint32_t source_primitive;
  uint32_t destination_point;
  uint32_t destination_primitive;

  BorderingPoint(uint32_t _source_point, uint32_t _source_primitive,  uint32_t _destination_point, uint32_t _destination_primitive) : 
    source_point(_source_point), source_primitive(_source_primitive), destination_point(_destination_point), destination_primitive(_destination_primitive) {}

} ;

struct Primitive {

    std::set<uint32_t> borders;    
    uint32_t first;
    uint32_t last;
    uint32_t nodes;
    int grain;

    Primitive(std::set<uint32_t> _borders): borders(_borders) {
        this->first =  *std::min_element(this->borders.begin(), this->borders.end());
        this->last  =  *std::max_element(this->borders.begin(), this->borders.end());
        this->nodes = this->last - this->first + 1;
        this->grain = (int)round(sqrt((float)nodes));        
        cout << "Grain: " << this->grain << std::endl;
    }

    boost::optional<uint32_t> index_up(uint32_t point_index) {
        uint32_t up_proposal = point_index - grain;
        if(this->belongs(up_proposal)){
            return up_proposal;
        } else {
            return {};
        }
    }

    boost::optional<uint32_t> index_down(uint32_t point_index) {
        uint32_t down_proposal = point_index + grain;
        if(this->belongs(down_proposal)){
            return down_proposal;
        } else {
            return {};
        }
    }

    boost::optional<uint32_t> index_right(uint32_t point_index) {
        if(point_index + 1 % this->grain == 0) {
            return {}; // right edge
        } else {
            return point_index + 1;
        }
    }

    boost::optional<uint32_t> index_left(uint32_t point_index) {
        if(point_index % this->grain == 0) {
            return {}; // left edge
        } else {
            return point_index - 1;
        }
    }

    bool belongs(uint32_t point_index) {
        return this->first <= point_index && point_index <= this->last;
    }

};

boost::optional<std::string> get_ply_path(int argc, char** argv) {
    const char* HELP = "help";
    const char* PATH = "path";
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        (HELP, "Usage: \n --path <PATH_TO_PLY_FILE>")
        (PATH, po::value<std::string>(), "path to PLY file")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
    if (vm.count(HELP) || !vm.count(PATH)) {
        std::cout << desc << std::endl;
        return {};
    } else {
        auto ply_file = vm[PATH].as<std::string>();
        std::cout << "Using PLY file: " << ply_file << std::endl;
        return ply_file;
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

pcl::PolygonMesh::Ptr load_mesh(std::string ply_path) {
    pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh());
    pcl::io::loadPolygonFilePLY(ply_path, *mesh);
    return mesh;
}

struct MeshTraits {
    typedef pcl::PointXYZ         VertexData;
    typedef pcl::geometry::NoData HalfEdgeData;
    typedef u_int32_t             EdgeData;
    typedef pcl::Normal           FaceData;
    typedef boost::false_type IsManifold;
};

typedef pcl::geometry::PolygonMesh <MeshTraits> Mesh;

void generate_mesh(pcl::PolygonMesh &polygon_mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud,  
        Mesh& mesh, Mesh::VertexIndices& vertex_indices) {
    for(pcl::PointXYZ const& vertex : point_cloud.points) {
       vertex_indices.push_back(mesh.addVertex (vertex));
    }
    for(pcl::Vertices const& polygon : polygon_mesh.polygons) {
        Mesh::VertexIndices face;
        for(uint32_t point_index : polygon.vertices) {
            face.push_back(vertex_indices[point_index]);
        }
        mesh.addFace(face);
    }
}

void extract_point_index_to_normals(pcl::PolygonMesh &polygon_mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud,    
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals) {
    Mesh::VertexIndices vertex_indices;            
    Mesh mesh;
    generate_mesh(polygon_mesh, point_cloud, mesh, vertex_indices);
    for(uint32_t point_index = 0 ; point_index < vertex_indices.size() ; ++point_index) {
        auto outgoing_half_edges = mesh.getOutgoingHalfEdgeAroundVertexCirculator(vertex_indices[point_index]);
        auto outgoing_half_edges_end = outgoing_half_edges;
        boost::optional<Eigen::Vector3f> previous_vector = {};
        do {
            Mesh::HalfEdgeIndex target_edge = outgoing_half_edges.getTargetIndex();
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

void extract_point_index_to_neighbours(pcl::PolygonMesh &polygon_mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud,    
        std::shared_ptr<std::map<uint32_t, std::vector<int>>> point_index_to_neighbours) {
    Mesh::VertexIndices vertex_indices;            
    Mesh mesh;
    generate_mesh(polygon_mesh, point_cloud, mesh, vertex_indices);
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
            uint8_t r = 255, g = 0, b = 0;
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

float non_border_curvature(Primitive primitive, int point_index, std::vector<int> neighbour_indices, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals_ptr) {
    /**
     * If it has 6 neighbours the orthogonal basis of the tangent space is the cross on U-V ignoring the two diagonal connections
     * 
     *    *   O   
     *      \ |  
     *    O---P---O
     *        | \
     *        O   *
     */
    pcl::PointCloud<pcl::PointXYZRGBNormal> &normals = *normals_ptr;    
    // normal-points
    pcl::PointXYZRGBNormal center  = normals[*primitive.index_down(point_index)];
    pcl::PointXYZRGBNormal down  = normals[*primitive.index_down(point_index)];
    pcl::PointXYZRGBNormal up    = normals[*primitive.index_up(point_index)];
    pcl::PointXYZRGBNormal left  = normals[*primitive.index_left(point_index)];
    pcl::PointXYZRGBNormal right = normals[*primitive.index_right(point_index)];
    // points
    Eigen::Vector3f center_point(center.x, center.y, center.z);
    Eigen::Vector3f down_point(down.x, down.y, down.z);
    Eigen::Vector3f up_point(up.x, up.y, up.z);
    Eigen::Vector3f left_point(left.x, left.y, left.z);
    Eigen::Vector3f right_point(right.x, right.y, right.z);
    //normals
    Eigen::Vector3f center_normal(center.normal_x, center.normal_y, center.normal_z);
    Eigen::Vector3f down_normal(down.normal_x, down.normal_y, down.normal_z);
    Eigen::Vector3f up_normal(up.normal_x, up.normal_y, up.normal_z);
    Eigen::Vector3f left_normal(left.normal_x, left.normal_y, left.normal_z);
    Eigen::Vector3f right_normal(right.normal_x, right.normal_y, right.normal_z);
    
    std::vector<float> curvatures;
    // radius on U
    auto center_left_norm =  (center_point - left_point ).norm();
    if(center_left_norm > 1e-3) {
        auto radius_left_center  =  (center_normal - left_normal  ).norm() / center_left_norm;
        curvatures.push_back(radius_left_center);
    }
    auto center_right_norm =  (center_point - right_point).norm();
    if(center_right_norm > 1e-3) {
        auto radius_right_center =  (right_normal  - center_normal).norm() / center_right_norm;
        curvatures.push_back(radius_right_center);
    }
    std::vector<float> curvatures_v;
    // radius on V
    auto center_up_norm = (center_point - up_point ).norm();
    if(center_up_norm > 1e-3) {
        auto radius_up_center  =  (center_normal - up_normal  ).norm() / center_up_norm;
        curvatures.push_back(radius_up_center);
    }
    auto center_down_norm = (center_point - down_point).norm();
    if(center_down_norm > 1e-3) {
        auto radius_down_center =  (down_normal  - center_normal).norm() / center_down_norm;
        curvatures.push_back(radius_down_center);
    }
    float sum_curvatures = 0.0;
    for(auto curvature : curvatures) {
        sum_curvatures = sum_curvatures + curvature;
    }
    auto mean_curvature = sum_curvatures / (float)curvatures.size();
    return mean_curvature;

}

void upper_left_and_bottom_right_corners(std::vector<int> indices, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals) {
    /**
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
     */ 

}

void bottom_left_and_upper_right_corners(std::vector<int> indices, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals) {
    /**
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

}


void non_cornering_borders(std::vector<int> indices, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals) {
    /**
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
     */ 

}

void extract_curvature(std::vector<Primitive> &primitives, 
        pcl::PolygonMesh &mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud, 
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &normals)  {
    std::shared_ptr<std::map<uint32_t, std::vector<int>>> point_index_to_neighbours (new std::map<uint32_t, std::vector<int>>);
    cout << "Starting curvature extraction..." << std::endl; 
    extract_point_index_to_neighbours(mesh, point_cloud, point_index_to_neighbours);
    
    for(auto primitive : primitives) {
        for(uint32_t point_index = primitive.first ; point_index <= primitive.last ; ++point_index) {
            auto neighbours = point_index_to_neighbours->find(point_index);
            if(neighbours != point_index_to_neighbours->end()) {
                auto point_and_neighbours = *neighbours;
                if(point_and_neighbours.second.size() == 6) {
                    float curvature = non_border_curvature(primitive, point_and_neighbours.first, point_and_neighbours.second, normals);
                    cout << curvature << std::endl;
                    curvature = curvature / 5;
                    curvature = curvature * curvature;
                    int normalized_curvature =(int) curvature;
                    if(normalized_curvature > 255){
                        normalized_curvature = 255;
                    }
                    uint8_t r = 0, g = normalized_curvature, b = 255 - normalized_curvature;
                    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                    (*normals)[point_and_neighbours.first].rgb = *reinterpret_cast<float*>(&rgb);
                }
            }
        }
    }
    cout << "... end of curvature extraction." << std::endl;     
}


pcl::visualization::PCLVisualizer::Ptr visualize(pcl::PolygonMesh::Ptr mesh,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr normals) {

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Mesh Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBNormal> rgb(normals);
    viewer->addPointCloud<pcl::PointXYZRGBNormal> (normals, rgb, "Current cloud point");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 3, "Current cloud point");
    viewer->addPointCloudNormals<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> (normals, normals, 10, 0.05, "Current normals");
    viewer->addPolygonMesh(*mesh);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    pcl::visualization::Camera camera;
    viewer->getCameraParameters(camera);
    camera.pos[2] = -1.5;
    viewer->setCameraParameters(camera);
    cout << "Camera: " << std::endl;
    cout << "\tFocal: "    << camera.focal[0] << ", " << camera.focal[1] << ", " << camera.focal[2] << std::endl;
    cout << "\tPosition: " << camera.pos[0] << ", " << camera.pos[1] << ", " << camera.pos[2]    << std::endl;
    cout << "\tView: "     << camera.view[0] << ", " << camera.view[1] << ", " << camera.view[2] << std::endl;
    cout << "\tClip: "     << camera.clip[0] << ", " << camera.clip[1] << std::endl;
    cout << "\tFovy: "     << camera.fovy  << std::endl;
    return viewer;
}

std::vector<Primitive> read_rows(std::istream& f) {
    std::string line;
    std::vector<Primitive> rows;
    while (std::getline(f, line)) {
        std::string entry;
        std::istringstream linestrm(line);
        auto borders = std::set<uint32_t>();
        while (std::getline(linestrm, entry, '\t')) {
            borders.insert(std::stoi(entry));
        }
        rows.push_back(Primitive(borders));
    }
    return rows;
}

boost::optional<std::vector<Primitive>> read_primitive_borders(std::string filename) {
    std::ifstream file(filename);
    if ( file ) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::vector<Primitive> rows = read_rows(buffer);
        file.close();
        return rows;
    } 
    return {};
}

void invert_normals(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals, 
    std::vector<Primitive> &primitive_borders,  uint32_t primitive_border_index) {
    auto current_primitive_borders = primitive_borders[primitive_border_index];
    for(uint32_t vertex_index = current_primitive_borders.first ; vertex_index <= current_primitive_borders.last ; ++vertex_index) {
        pcl::PointXYZRGBNormal& point_normal = (*output_normals)[vertex_index];
        point_normal.normal_x = - point_normal.normal_x;
        point_normal.normal_y = - point_normal.normal_y;
        point_normal.normal_z = - point_normal.normal_z;
    }
}

boost::optional<uint32_t> get_primitive_border_index(uint32_t point_index, std::vector<Primitive> &primitive_borders) {
    for (uint32_t primitive_border_index = 0; primitive_border_index < primitive_borders.size(); ++primitive_border_index) {
        auto current_primitive_borders = primitive_borders[primitive_border_index];
        if(current_primitive_borders.belongs(point_index)) {        
            return primitive_border_index;
        }
    }
    return {};
}

void detect_bordering_points(
            pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, 
            std::vector<Primitive> &primitive_borders,
            std::map<uint32_t, std::vector<BorderingPoint>> &bordering_points) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(point_cloud);
    pcl::PointXYZ searchPoint;
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for(uint32_t border_index = 0 ; border_index < primitive_borders.size() ; ++border_index) {
        std::set<uint32_t> borders = primitive_borders[border_index].borders;
        // for each border get the closest node such that it's not in the border
        for(uint32_t const& border_point : borders) {
            pcl::PointXYZ point_in_border = (*point_cloud)[border_point];
            auto total_found = kdtree.nearestKSearch(point_in_border, K, pointIdxNKNSearch, pointNKNSquaredDistance);
            if(total_found  > 0) {
                bool found = false;
                for(uint32_t current_neighbour = 0 ; !found && current_neighbour < total_found ; current_neighbour++) {                
                    auto current_index = pointIdxNKNSearch[current_neighbour];
                    auto primitive_of_index_opt = get_primitive_border_index(current_index, primitive_borders);                    
                    if(primitive_of_index_opt){                        
                        found = *primitive_of_index_opt != border_index;
                        if(found){
                            auto current_bordering_point = BorderingPoint(border_point, border_index, current_index, *primitive_of_index_opt);
                            if(bordering_points.find(border_index) == bordering_points.end()){
                                bordering_points[border_index] = std::vector<BorderingPoint>();
                            }
                            bordering_points[border_index].push_back(current_bordering_point);
                        }
                    }
                }
            }
        }
    }
}

boost::optional<uint32_t> viewpoint_normal_consistency(pcl::PointXYZ viewpoint,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud,
        std::vector<Primitive> &primitive_borders,
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(point_cloud);
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    if ( kdtree.nearestKSearch(viewpoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
    {
        size_t closest_to_viewpoint = pointIdxNKNSearch[0];
        auto border_index_opt = get_primitive_border_index(closest_to_viewpoint, primitive_borders);
        if(border_index_opt) {
            uint32_t border_index = *border_index_opt;
            cout << "Primitive index of point closest to viewpoint: " << border_index << std::endl;
            pcl::PointXYZRGBNormal closest_to_viewpoint_pointnormal = (*output_normals)[closest_to_viewpoint];
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

void make_smooth_manifold(pcl::PolygonMesh &polygon_mesh, 
            pcl::PointCloud<pcl::PointXYZ>::Ptr &point_cloud, 
            std::vector<Primitive> &primitive_borders, 
            pcl::PointXYZ viewpoint,
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr &output_normals) {
    std::map<uint32_t, std::vector<BorderingPoint>> bordering_points;
    detect_bordering_points(point_cloud, primitive_borders, bordering_points);
    cout << "Bordering points: " << bordering_points.size() << std::endl;
    Mesh::VertexIndices vertex_indices;            
    Mesh mesh;
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
                std::vector<BorderingPoint> bordering_points = (*bordering_points_interator).second;
                //cout << "Total number of borders: " <<  bordering_points.size() << std::endl;
                for(BorderingPoint bordering_point : bordering_points) {
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

int main (int argc, char** argv)
{
    auto ply_path_opt = get_ply_path(argc, argv);
    auto primitive_borders_opt = read_primitive_borders("plane_input_demo.png30000.tsv");
    if(ply_path_opt && primitive_borders_opt) {
        std::vector<Primitive> primitive_borders = *primitive_borders_opt;
        cout << "Number of primitives: " << primitive_borders.size() << std::endl;
        std::string ply_path = *ply_path_opt;
        pcl::PolygonMesh::Ptr smoothed_mesh = load_mesh(ply_path);
        //pcl::PolygonMesh::Ptr smoothed_mesh = laplacian_smoothing(mesh);
        //pcl::io::savePolygonFilePLY("smoothed", *smoothed_mesh);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals_ptr (new std::map<uint32_t, std::vector<Eigen::Vector3f>>);
        pcl::fromPCLPointCloud2(smoothed_mesh->cloud, *point_cloud_ptr);
        extract_normals(*smoothed_mesh, *point_cloud_ptr, output_normals);
        extract_curvature(primitive_borders, *smoothed_mesh, *point_cloud_ptr, output_normals);

        pcl::toPCLPointCloud2(*output_normals, smoothed_mesh->cloud);
        cout << "Number of points: "  << point_cloud_ptr->size() << std::endl;
        cout << "Number of normals: " << output_normals->size()  << std::endl;
        pcl::PointXYZ viewpoint(0, 0, -1.5);
        make_smooth_manifold(*smoothed_mesh, point_cloud_ptr, primitive_borders, viewpoint, output_normals);
        
        //pcl::Poisson<pcl::PointXYZRGBNormal> poisson;
        //poisson.setDepth(12);
        //poisson.setInputCloud(output_normals);
        //poisson.reconstruct(*smoothed_mesh);

        auto viewer = visualize(smoothed_mesh, output_normals);
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }

        return 0;
    } else {
        return 1;
    }
}
