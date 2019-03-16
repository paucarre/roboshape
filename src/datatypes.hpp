#pragma once

// pcl
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/conversions.h>
#include <pcl/surface/poisson.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/processing.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_smoothing_laplacian.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
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
#include <stack>
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
                return {}; 
            } else {
                return point_index + 1;
            }
        }

        boost::optional<uint32_t> index_left(uint32_t point_index) {
            if(point_index % this->grain == 0) {
                return {}; 
            } else {
                return point_index - 1;
            }
        }

        bool belongs(uint32_t point_index) {
            return this->first <= point_index && point_index <= this->last;
        }

    };

    std::vector<Primitive> generate_primitives(int number_of_primitives, int number_of_proints) {
        int grain = ((int)sqrt(number_of_proints / number_of_primitives )) - 1;
        std::vector<Primitive> primitives;
        for(int primitive_index = 0 ; primitive_index < number_of_primitives ; primitive_index++){
            int offset = ( grain + 1 ) * ( grain + 1 ) * primitive_index;
            std::set<uint32_t> borders;
            for(int step = 0 ; step < grain + 1 ; step++) {
                int first_column = offset + ( ( grain + 1 ) * step);
                int last_column  = offset + grain + ( ( grain + 1 ) * step);
                int first_row    = offset + step;
                int last_row     = offset + step + ( ( grain + 1 ) * grain );
                borders.insert(first_column);
                borders.insert(last_column);
                borders.insert(first_row);
                borders.insert(last_row);
            }
            primitives.push_back(Primitive(borders));
        }
        return primitives;
    }

    boost::optional<uint32_t> get_primitive_border_index(uint32_t point_index, std::vector<roboshape::Primitive> &primitive_borders) {
        for (uint32_t primitive_border_index = 0; primitive_border_index < primitive_borders.size(); ++primitive_border_index) {
            auto current_primitive_borders = primitive_borders[primitive_border_index];
            if(current_primitive_borders.belongs(point_index)) {        
                return primitive_border_index;
            }
        }
        return {};
    }

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