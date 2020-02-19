#pragma once

#include "integrand_typedefs.h"
#include <vector>
#include "SnS_Scenario.h"

void gauleg(const double x1, const double x2, std::vector<double>& x, std::vector<double>& w);
AssemblyFunction MakeMatrix(SnS_Scenario const& scenario, IntegrandBlockIJ const& fa, Eigen::SparseMatrix<double>& a);
AssemblyFunction MakeMatrixConsistent(SnS_Scenario const& scenario, IntegrandBlockIJ const& fa, Eigen::SparseMatrix<double>& a);
AssemblyFunction MakeMatrixLumped(SnS_Scenario const& scenario, IntegrandBlockIJ const& fa, Eigen::SparseMatrix<double>& a);
AssemblyFunction MakeVectorBlockI(SnS_Scenario const& scenario, IntegrandBlockI const& fb, Eigen::VectorXd& b);
SurfaceAssemblyFunction MakeVectorBlockI(SnS_Scenario const& scenario, SurfaceIntegrandBlockI const& fb, Eigen::VectorXd& b);
