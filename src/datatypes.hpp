#pragma once

// pcl
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


// boost
#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

// std
#include <set>
#include <tgmath.h> 
#include <numeric>  
#include <iostream>

namespace roboshape { 

    struct MeshTraits {
        typedef pcl::PointXYZ         VertexData;
        typedef pcl::geometry::NoData HalfEdgeData;
        typedef u_int32_t             EdgeData;
        typedef pcl::Normal           FaceData;
        typedef boost::false_type IsManifold;
    };

    typedef pcl::geometry::PolygonMesh <MeshTraits> Mesh;

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

}