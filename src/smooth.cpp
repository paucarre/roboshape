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


#include <numeric>  
#include <iostream>

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

Eigen::Vector3f extract_normal(pcl::Vertices const& polygon, pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Vector3f &center) {
    auto point_1 = point_cloud[polygon.vertices[0]];
    auto point_2 = point_cloud[polygon.vertices[1]];
    auto point_3 = point_cloud[polygon.vertices[2]];
    Eigen::Vector3f point_1_eig(point_1.x, point_1.y, point_1.z);
    Eigen::Vector3f point_2_eig(point_2.x, point_2.y, point_2.z);
    Eigen::Vector3f point_3_eig(point_3.x, point_3.y, point_3.z);
    
    Eigen::Vector3f vector_1_2 = point_2_eig - point_1_eig;
    Eigen::Vector3f vector_1_3 = point_3_eig - point_1_eig;
    Eigen::Vector3f vector_normal = vector_1_2.cross(vector_1_3); 
    vector_normal.normalize();

    Eigen::Vector3f centroid = (point_1_eig + point_2_eig + point_3_eig) / 3.0;
    if((centroid - center).adjoint() * vector_normal < 0.0 ){
        vector_normal = -vector_normal;
    }
    return vector_normal;
}

Eigen::Vector3f compute_center(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for(pcl::PointXYZ point : point_cloud->points) {
        Eigen::Vector3f point_eigen(point.x, point.y, point.z);
        sum = sum + point_eigen;
    }    
    return sum / point_cloud->points.size();
}

struct MeshTraits {
    typedef pcl::PointXYZ         VertexData;
    typedef pcl::geometry::NoData HalfEdgeData;
    typedef u_int32_t             EdgeData;
    typedef pcl::Normal           FaceData;
    typedef boost::false_type IsManifold;
};
typedef pcl::geometry::PolygonMesh <MeshTraits> Mesh;

void printEdge (const Mesh& mesh, const Mesh::HalfEdgeIndex& idx_he) {
  std::cout << "  "
            << mesh.getVertexDataCloud () [mesh.getOriginatingVertexIndex (idx_he).get ()]
            << " "
            << mesh.getVertexDataCloud () [mesh.getTerminatingVertexIndex (idx_he).get ()]
            << std::endl;
}

void extract_point_index_to_normals(pcl::PolygonMesh &polygon_mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud,
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals) {
    Mesh mesh;
    Mesh::VertexIndices vertex_indices;
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
                Eigen::Vector3f normal = -(*previous_vector).cross(current_vector); 
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

void extract_point_index_to_normals_old(pcl::PolygonMesh &mesh, Eigen::Vector3f &center, 
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> &point_index_to_normals) {
    pcl::PointCloud<pcl::PointNormal>::Ptr outputCloud(new pcl::PointCloud<pcl::PointNormal>);     
    pcl::PointCloud<pcl::PointXYZ> point_cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, point_cloud);
    for(pcl::Vertices const& polygon : mesh.polygons) {
        Eigen::Vector3f normal = extract_normal(polygon, point_cloud, center);
        for(uint32_t point_index : polygon.vertices) {
            auto vertices_iterator = point_index_to_normals->find(point_index);
            if (vertices_iterator == point_index_to_normals->end()) { // not found
                (*point_index_to_normals)[point_index] = { normal };
            } else {
                vertices_iterator->second.push_back(normal);
            }
        }
    }           
}

void extract_normals(pcl::PolygonMesh &mesh, pcl::PointCloud<pcl::PointXYZ> &point_cloud, 
        pcl::PointCloud<pcl::Normal>::Ptr &output_normals)  {
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
            pcl::Normal normal = pcl::Normal();
            normal.normal_x = sum_of_normals[0];
            normal.normal_y = sum_of_normals[1];
            normal.normal_z = sum_of_normals[2];
            output_normals->points.push_back(normal);            
        } else {
            //cout << "WARNING: point without normal found." << std::endl;     
        }
    }
    cout << "... end of normal extraction." << std::endl;     
}

pcl::visualization::PCLVisualizer::Ptr visualize(pcl::PolygonMesh::Ptr mesh,
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, 
        pcl::PointCloud<pcl::Normal>::ConstPtr normals) {
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Mesh Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZ> rgb(cloud);
    // viewer->addPointCloud<pcl::PointXYZ> (cloud, rgb, "Current cloud point");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Current cloud point");
    viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 10, 0.05, "Current normals");
    viewer->addPolygonMesh(*mesh);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return viewer;
}

int main (int argc, char** argv)
{
    auto ply_path_opt = get_ply_path(argc, argv);
    if(ply_path_opt) {
        auto ply_path = *ply_path_opt;
        pcl::PolygonMesh::Ptr mesh = load_mesh(ply_path);
        pcl::PolygonMesh::Ptr smoothed_mesh = laplacian_smoothing(mesh);
        pcl::io::savePolygonFilePLY("smoothed", *smoothed_mesh);
        pcl::PointCloud<pcl::Normal>::Ptr output_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(smoothed_mesh->cloud, *point_cloud_ptr);        
        //auto center = compute_center(point_cloud_ptr);
        extract_normals(*smoothed_mesh, *point_cloud_ptr, output_normals);
        //std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals(new std::map<uint32_t, std::vector<Eigen::Vector3f>>);
        //test(*smoothed_mesh, *point_cloud_ptr, point_index_to_normals);
        cout << "Number of points: "  << point_cloud_ptr->size() << std::endl;
        cout << "Number of normals: " << output_normals->size()  << std::endl;
        auto viewer = visualize(smoothed_mesh, point_cloud_ptr, output_normals);
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }
        //pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
        //pcl::fromPCLPointCloud2(mesh->cloud, *point_cloud);
        //pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        //pcl::PointCloud<pcl::PointNormal> mls_points;
        //PointCloud<PointNormal>::Ptr cloud_smoothed_normals(new PointCloud<PointNormal>());
        //pcl::Poisson<pcl::PointNormal> poisson;
        //poisson.setDepth(9);
        //poisson.setInputCloud(cloud_smoothed_normals);
        //PolygonMesh poisson_mesh;
        //poisson.reconstruct(poisson_mesh);
    
        //pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
        //mls.setComputeNormals (true);
        //mls.setInputCloud (point_cloud);
        //mls.setPolynomialOrder (2);
        //mls.setSearchMethod (tree);
        //mls.setSearchRadius (0.03);
        //mls.process (mls_points);
        //pcl::io::savePolygonFilePLY ("result-mls.PLY", mls_points);
        return 0;
    } else {
        return 1;
    }
}
