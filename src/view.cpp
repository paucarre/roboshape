// roboshape
#include "curvature.hpp"
#include "surface.hpp"

// Boost library
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
        return {};
    } else {
        auto ply_file = vm[PATH].as<std::string>();
        std::cout << "Using PLY file: " << ply_file << std::endl;
        return ply_file;
    }
}


pcl::PolygonMesh::Ptr load_mesh(std::string ply_path) {
    pcl::PolygonMesh::Ptr mesh (new pcl::PolygonMesh());
    pcl::io::loadPolygonFilePLY(ply_path, *mesh);
    return mesh;
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

std::vector<roboshape::Primitive> read_rows(std::istream& f) {
    std::string line;
    std::vector<roboshape::Primitive> rows;
    while (std::getline(f, line)) {
        std::string entry;
        std::istringstream linestrm(line);
        auto borders = std::set<uint32_t>();
        while (std::getline(linestrm, entry, '\t')) {
            borders.insert(std::stoi(entry));
        }
        rows.push_back(roboshape::Primitive(borders));
    }
    return rows;
}

boost::optional<std::vector<roboshape::Primitive>> read_primitive_borders(std::string filename) {
    std::ifstream file(filename);
    if ( file ) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::vector<roboshape::Primitive> rows = read_rows(buffer);
        file.close();
        return rows;
    } 
    return {};
}


int main (int argc, char** argv)
{
    auto ply_path_opt = get_ply_path(argc, argv);
    auto primitive_borders_opt = read_primitive_borders("plane_input_demo.png30000.tsv");
    if(ply_path_opt && primitive_borders_opt) {
        std::vector<roboshape::Primitive> primitive_borders = *primitive_borders_opt;
        cout << "Number of primitives: " << primitive_borders.size() << std::endl;
        std::string ply_path = *ply_path_opt;
        pcl::PolygonMesh::Ptr mesh = load_mesh(ply_path);
    
        pcl::PolygonMesh::Ptr smoothed_mesh = roboshape::laplacian_smoothing(mesh);
        pcl::io::savePolygonFilePLY("smoothed", *smoothed_mesh);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals_ptr (new std::map<uint32_t, std::vector<Eigen::Vector3f>>);
        pcl::fromPCLPointCloud2(smoothed_mesh->cloud, *point_cloud_ptr);
        roboshape::extract_normals(*smoothed_mesh, *point_cloud_ptr, output_normals);
        pcl::PointXYZ viewpoint(0, 0, -1.5);
        roboshape::make_smooth_manifold(*smoothed_mesh, point_cloud_ptr, primitive_borders, viewpoint, output_normals);
        extract_planes(primitive_borders, *smoothed_mesh, *point_cloud_ptr, output_normals);
        pcl::toPCLPointCloud2(*output_normals, smoothed_mesh->cloud);
        cout << "Number of points: "  << point_cloud_ptr->size() << std::endl;
        cout << "Number of normals: " << output_normals->size()  << std::endl;
        
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
