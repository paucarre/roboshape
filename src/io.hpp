#pragma once


// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/visualization/pcl_visualizer.h>

// boost
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

// std
#include <set>
#include <tgmath.h> 

// boost
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

// std
#include <iostream>

namespace roboshape {

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

}

