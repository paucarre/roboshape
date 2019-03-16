
#include "curvature.hpp"
#include "surface.hpp"
#include "io.hpp"


int main (int argc, char** argv)
{
    auto ply_path_opt = roboshape::get_ply_path(argc, argv);
    if(ply_path_opt) {
        std::string ply_path = *ply_path_opt;
        pcl::PolygonMesh::Ptr mesh = roboshape::load_mesh(ply_path);
        std::vector<roboshape::Primitive> primitive_borders = roboshape::generate_primitives(25, mesh->cloud.data.size());
        cout << "Number of primitives: " << primitive_borders.size() << std::endl;  
        pcl::PolygonMesh::Ptr smoothed_mesh = roboshape::laplacian_smoothing(mesh);
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr output_normals (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
        std::shared_ptr<std::map<uint32_t, std::vector<Eigen::Vector3f>>> point_index_to_normals_ptr (new std::map<uint32_t, std::vector<Eigen::Vector3f>>);
        pcl::fromPCLPointCloud2(smoothed_mesh->cloud, *point_cloud_ptr);
        roboshape::extract_normals(*smoothed_mesh, *point_cloud_ptr, output_normals);
        pcl::PointXYZ viewpoint(0, 0, -1.5);
        roboshape::make_smooth_manifold(*smoothed_mesh, point_cloud_ptr, primitive_borders, viewpoint, output_normals);
        roboshape::extract_planes(primitive_borders, *point_cloud_ptr, *smoothed_mesh, output_normals);
        cout << "Number of points: "  << point_cloud_ptr->size() << std::endl;
        cout << "Number of normals: " << output_normals->size()  << std::endl;
        
        auto viewer = roboshape::visualize(smoothed_mesh, output_normals);
        while (!viewer->wasStopped ()){
            viewer->spinOnce (100);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }

        return 0;
    } else {
        return 1;
    }
}
