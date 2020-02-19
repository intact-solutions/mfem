#include "gauleg.h"
#include <stdio.h>
#include <iostream>
using namespace Eigen;

#define PI 3.141592653589793238462643383279502884197

#include <math.h>
#define EPS 1.0e-14
void gauleg(const double x1, const double x2, std::vector<double>& x, std::vector<double>& w)
/*
  gauleg procedure is taken from the Numerical Recipes
  Purpose - computation of abscissae and weights of Gauss-Legendre formula
*/
// FIXME: 1-based indexing
{
  std::fill(x.begin(), x.end(), 0);
  std::fill(w.begin(), w.end(), 0);
  int n = (int)x.size() - 1;
  double z1, z, pp, p3, p2, p1;

  int m = (n + 1) / 2;
  double xm = 0.5 * (x2 + x1);
  double xl = 0.5 * (x2 - x1);
  for (int i = 1; i <= m; i++)
  {
    z = cos(PI * (i - 0.25) / (n + 0.5));
    do
    {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; j++)
      {
        p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / pp;
    } while (fabs(z - z1) > EPS);
    x[i] = xm - xl * z;
    x[n + 1 - i] = xm + xl * z;
    w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
    w[n + 1 - i] = w[i];
  }
}
#undef EPS

AssemblyFunction MakeMatrix(SnS_Scenario const& scenario, IntegrandBlockIJ const& fa, Eigen::SparseMatrix<double>& A) {
  return [&fa, &A, &scenario] (QuadraturePoint const& quad_point) -> void {
    auto const neighborhood = scenario.Neighborhood(quad_point);
    for (auto const& basis_i : neighborhood) {
      for (auto const& basis_j : neighborhood) {
        if (basis_i.m_global_index <= basis_j.m_global_index) {
          // avoid double-counting
          Matrix3d const value = fa(quad_point.position, basis_i, basis_j) * quad_point.weight;
          for (int solution_dimension_i = 0; solution_dimension_i < 3; solution_dimension_i++) {
            int start_dimension = basis_i.m_global_index == basis_j.m_global_index ? solution_dimension_i : 0;
            for (int solution_dimension_j = start_dimension; solution_dimension_j < 3; solution_dimension_j++) {
              unsigned long i = basis_i.components[solution_dimension_i].m_index;
              unsigned long j = basis_j.components[solution_dimension_j].m_index;
              if (i != j) {
                #pragma omp atomic
                A.coeffRef(i, j) += value(solution_dimension_i, solution_dimension_j);
                #pragma omp atomic
                A.coeffRef(j, i) += value(solution_dimension_i, solution_dimension_j);
              } else {
                #pragma omp atomic
                A.coeffRef(i, j) += value(solution_dimension_i, solution_dimension_j);
              }
            }
          }
        }
      }
    }
  };
}

AssemblyFunction MakeMatrixConsistent(SnS_Scenario const& scenario, IntegrandBlockIJ const& fa, Eigen::SparseMatrix<double>& A) {
  return [&fa, &A, &scenario](QuadraturePoint const& quad_point) -> void {
    auto const neighborhood = scenario.Neighborhood(quad_point);
    for (auto const& basis_i : neighborhood) {
      for (auto const& basis_j : neighborhood) {
        if (basis_i.m_global_index <= basis_j.m_global_index) {
          // avoid double-counting
          Matrix3d const value = fa(quad_point.position, basis_i, basis_j) * quad_point.weight;
          for (int solution_dimension_i = 0; solution_dimension_i < 3; solution_dimension_i++) {
            int start_dimension = basis_i.m_global_index == basis_j.m_global_index ? solution_dimension_i : 0;
            for (int solution_dimension_j = start_dimension; solution_dimension_j < 3; solution_dimension_j++) {
              unsigned long i = basis_i.components[solution_dimension_i].m_index;
              unsigned long j = basis_j.components[solution_dimension_j].m_index;
              if (i != j) {
                #pragma omp atomic
                A.coeffRef(i, j) += value(solution_dimension_i, solution_dimension_j);
                #pragma omp atomic
                A.coeffRef(j, i) += value(solution_dimension_i, solution_dimension_j);
              }
              else {
                #pragma omp atomic
                A.coeffRef(i, j) += value(solution_dimension_i, solution_dimension_j);
              }
            }
          }
        }
      }
    }
  };
}

AssemblyFunction MakeMatrixLumped(SnS_Scenario const& scenario, IntegrandBlockIJ const& fa, Eigen::SparseMatrix<double>& A) {
  return [&fa, &A, &scenario](QuadraturePoint const& quad_point) -> void {
    auto const neighborhood = scenario.Neighborhood(quad_point);
    for (auto const& basis_i : neighborhood) {
      for (auto const& basis_j : neighborhood) {
        if (basis_i.m_global_index <= basis_j.m_global_index) {
          // avoid double-counting
          Matrix3d const value = fa(quad_point.position, basis_i, basis_j) * quad_point.weight;
          for (int solution_dimension_i = 0; solution_dimension_i < 3; solution_dimension_i++) {
            int start_dimension = basis_i.m_global_index == basis_j.m_global_index ? solution_dimension_i : 0;
            for (int solution_dimension_j = start_dimension; solution_dimension_j < 3; solution_dimension_j++) {
              unsigned long i = basis_i.components[solution_dimension_i].m_index;
              unsigned long j = basis_j.components[solution_dimension_j].m_index;
              if (i == j) {
#pragma omp atomic
                A.coeffRef(i, j) += value(solution_dimension_i, solution_dimension_j);
              }
            }
          }
        }
      }
    }
  };
}

AssemblyFunction MakeVectorBlockI(SnS_Scenario const& scenario, IntegrandBlockI const& fb, Eigen::VectorXd& b) {
  return [&fb, &b, &scenario] (QuadraturePoint const& quad_point) -> void {
    auto const bases = scenario.Neighborhood(quad_point);
    for (auto const& basis : bases) {
      int solution_dimension = 0;
      vec3 const value = fb(quad_point.position, basis);
      for (BasisComponent const& component : basis.components) {
        #pragma omp atomic
        b[component.m_index] += value[solution_dimension] * quad_point.weight;
        solution_dimension++;
      }
    }
  };
}

SurfaceAssemblyFunction MakeVectorBlockI(SnS_Scenario const& scenario, SurfaceIntegrandBlockI const& fb, Eigen::VectorXd& b) {
  return [&fb, &b, &scenario] (SurfaceQuadraturePoint const& coord) -> void {
    auto const bases = scenario.Neighborhood(coord);
    double const w = coord.weight;
    for (auto const& basis : bases) {
      int solution_dimension = 0;
      vec3 const value = fb(coord, basis);
      for (BasisComponent const& component : basis.components) {
        #pragma omp atomic
        b[component.m_index] += value[solution_dimension] * w;
        solution_dimension++;
      }
    }
  };
}
