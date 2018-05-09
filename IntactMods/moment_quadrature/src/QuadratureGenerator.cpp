#include "../IntactMods/moment_quadrature/inc/QuadratureGenerator.h"
#include <random>

vector<MatrixInverse*> QuadratureGenerator::m_normalized_matrices;
vector<vector<array<double,3>>> QuadratureGenerator::m_normalized_points;
vector<Vector> QuadratureGenerator::m_normalized_weights;
void gauleg(const double x1, const double x2, std::vector<double>& x, std::vector<double>& w);

QuadratureGenerator::QuadratureGenerator(Vector moments, vec3 origin, vec3 cell_size, int order, bool is_boundary)
{
  m_order_function = order;
  m_point_count = (unsigned)pow(m_order_function + 1, 3);
  m_quad_weights = m_normalized_weights[m_order_function];
  m_quad_points = m_normalized_points[m_order_function];
  m_new_origin = origin;
  m_scale_factor = cell_size;

  if (is_boundary) {
    AdjustWeights(moments);
  } else {
    m_quad_weights = m_normalized_weights[m_order_function];
  }

  // full cells must also be rescaled!
  RescaleQuadPoints();
}

void QuadratureGenerator::InitializeNormalizedQuadratures(unsigned order)
{
  m_normalized_matrices.resize(order + 1);
  m_normalized_points.resize(order + 1);
  m_normalized_weights.resize(order + 1);

  for (unsigned order_function = 0; order_function <= order; order_function++) {
    unsigned point_count = pow(order_function + 1, 3);

    // allocate Gauss unit cube
    int len = order_function + 2; //[0] elements are to be ignored
    vector<double> x_gauss(len);
    vector<double> w_gauss(len);
    gauleg(-0.5, 0.5, x_gauss, w_gauss);

    vector<vec3> quad_points;
    quad_points.reserve(point_count);
    Vector quad_weights = Vector(point_count);
    int index = 0;
    //[0] elements are to be ignored
    for (int k = 1; k < len; k++) {
      for (int j = 1; j < len; j++) {
        for (int i = 1; i < len; i++) {
          vec3 point = { x_gauss[i], x_gauss[j], x_gauss[k] };
          quad_points.push_back(point);
          quad_weights(index) = w_gauss[i] * w_gauss[j] * w_gauss[k];
          index++;
        }
      }
    }

    DenseMatrix moment_fitting_matrix(point_count, point_count);
    for (unsigned column = 0; column < point_count; column++) {
      for (unsigned i = 0; i <= order_function; i++) {
        for (unsigned j = 0; j <= order_function; j++) {
          for (unsigned k = 0; k <= order_function; k++) {
            int row = i * (order_function + 1) * (order_function + 1) + j * (order_function + 1) + k;
            moment_fitting_matrix(row, column) = pow(quad_points[column][0], i) *
                                                 pow(quad_points[column][1], j) *
                                                 pow(quad_points[column][2], k);
          }
        }
      }
    }

    m_normalized_matrices.at(order_function) = moment_fitting_matrix.Inverse(); //not exactly inverse
    m_normalized_points.at(order_function) = quad_points;
    m_normalized_weights.at(order_function) = quad_weights;
  }
}

void QuadratureGenerator::AdjustWeights(Vector moment_vec)
{
  MatrixInverse* moment_fitting_matrix = m_normalized_matrices[m_order_function];
  moment_fitting_matrix->Mult(moment_vec, m_quad_weights);
}

/*Transforms quadrature points from being for a unit/normalized cell to the current cell.
Impporant! assume 1x1x1 unit cell centered at origin.
*/
void QuadratureGenerator::RescaleQuadPoints()
{
  for (unsigned i = 0; i < m_point_count; i++) {
    for (int j = 0; j < 3; j++) {
      //scale up and translate
      m_quad_points[i][j] = m_quad_points[i][j] * m_scale_factor[j];
      m_quad_points[i][j] = m_quad_points[i][j] + m_new_origin[j];
    }
  }
  double volume = m_scale_factor[0] * m_scale_factor[1] * m_scale_factor[2];
  for (unsigned j = 0; j < m_point_count; j++) {
    m_quad_weights[j] = m_quad_weights[j] * volume;
  }
}

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

