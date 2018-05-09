#pragma once
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
using namespace mfem;
using namespace std;
typedef array<double, 3> vec3;

class QuadratureGenerator
{
public:
  QuadratureGenerator(Vector moments, vec3 origin, vec3 cell_size, int order, bool is_boundary);
  static void InitializeNormalizedQuadratures(unsigned order);
  std::vector<array<double, 3>> m_quad_points;
  Vector m_quad_weights;
private:
  static vector<MatrixInverse*> m_normalized_matrices;
  static vector<vector<array<double, 3>>> m_normalized_points;
  static vector<Vector> m_normalized_weights;
  unsigned int m_point_count;
  vec3 m_new_origin, m_scale_factor;
  // moment fitting variables and method
  void RescaleQuadPoints();
  void AdjustWeights(Vector moments);
  unsigned int m_order_function;
};