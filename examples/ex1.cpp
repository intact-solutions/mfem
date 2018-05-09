//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include "../IntactMods/tinyply/source/tinyply.h"
#include "../IntactMods/json/single_include/nlohmann/json.hpp"
#include "../IntactMods/moment_quadrature/inc/QuadratureGenerator.h"

using namespace std;
using namespace mfem;
using json = nlohmann::json;

struct MomentCell {
  IndexTriplet index;
  vec3 scale_factor, origin;
  unsigned order;
  vector<double> moments;
  bool is_boundary = false;
};

vector<MomentCell> cells;
int global_element_idx = 0;

//class MyCoefficient;


void ReadMoments(string moment_filename);
void GetMFIntegrationRule(IntegrationRule& intrule, MomentCell cell);

class MyCoefficient : public Coefficient
{
private:
  GridFunction & u;
  ConstantCoefficient lambda, mu;
  DenseMatrix eps, sigma;

public:
  MyCoefficient(GridFunction &_u, ConstantCoefficient &_lambda, ConstantCoefficient &_mu)
    : u(_u), lambda(_lambda), mu(_mu) { }
  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
  {
    u.GetVectorGradient(T, eps);  // eps = grad(u)
    eps.Symmetrize();             // eps = (1/2)*(grad(u) + grad(u)^t)
    double l = lambda.Eval(T, ip);
    double m = mu.Eval(T, ip);
    sigma.Diag(l*eps.Trace(), eps.Size()); // sigma = lambda*trace(eps)*I
    sigma.Add(2 * m, eps);          // sigma += 2*mu*eps
    //cout << "size of stress matrix" << sigma.Size();
    double sigdif1 = (sigma(0, 0) - sigma(1, 1));
    double sigdif2 = (sigma(2, 2) - sigma(1, 1));
    double sigdif3 = (sigma(0, 0) - sigma(2, 2));
    double sigshearsq =
      sigma(0, 1)*sigma(0, 1) +
      sigma(0, 2)*sigma(0, 2) +
      sigma(2, 1)*sigma(2, 1);
    double vonmises = sqrt(sigdif1*sigdif1 + sigdif2 * sigdif2 + sigdif3 * sigdif3 + 6 * sigshearsq);
    return vonmises; // return sigma_xx
  }
  virtual void Read(istream &in) { }
  virtual ~MyCoefficient() { }
};

/** Integrator for the linear elasticity form:
a(u,v) = (lambda div(u), div(v)) + (2 mu e(u), e(v)),
where e(v) = (1/2) (grad(v) + grad(v)^T).
This is a 'Vector' integrator, i.e. defined for FE spaces
using multiple copies of a scalar FE space. */
class ElasticityIntegratorCustom : public BilinearFormIntegrator
{
private:
  double q_lambda, q_mu;
  Coefficient *lambda, *mu;

#ifndef MFEM_THREAD_SAFE
  DenseMatrix dshape, //dshape is the gradient of the shape function in ref space
    Jinv,   //Jacobian inverse
    gshape, //gspace seems to be the gradient in the geometry? space
    pelmat; //gspace \dot gspace^T

  Vector divshape;
#endif

public:
  ElasticityIntegratorCustom(Coefficient &l, Coefficient &m)
  {
    lambda = &l; mu = &m;
  }
  /** With this constructor lambda = q_l * m and mu = q_m * m;
  if dim * q_l + 2 * q_m = 0 then trace(sigma) = 0. */
  ElasticityIntegratorCustom(Coefficient &m, double q_l, double q_m)
  {
    lambda = NULL; mu = &m; q_lambda = q_l; q_mu = q_m;
  }

  virtual void AssembleElementMatrix(
    const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
  {
    int dof = el.GetDof();
    int dim = el.GetDim();
    double w, L, M;

#ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape(dof, dim), Jinv(dim), gshape(dof, dim), pelmat(dof);
    Vector divshape(dim*dof);
#else
    Jinv.SetSize(dim);
    dshape.SetSize(dof, dim);
    gshape.SetSize(dof, dim);
    pelmat.SetSize(dof);
    divshape.SetSize(dim*dof);
#endif

    elmat.SetSize(dof * dim);

    IntegrationRule ir;

    //if (ir == NULL)
    //{
    //  int order = 2 * Trans.OrderGrad(&el); // correct order?
    //  ir = IntRules.Get(el.GetGeomType(), order);
    //}

    GetMFIntegrationRule(ir, cells[global_element_idx++]);

    elmat = 0.0;
    double total_w = 0;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
      const IntegrationPoint &ip = ir.IntPoint(i);
      total_w += ip.weight;
      //cout << "Integration point x,y,z: [" << ip.x << ", " << ip.y << ", " << ip.z << "]" << "w: " << total_w << "\n";

      el.CalcDShape(ip, dshape);

      Trans.SetIntPoint(&ip);
      w = ip.weight * Trans.Weight();
      CalcInverse(Trans.Jacobian(), Jinv);
      Mult(dshape, Jinv, gshape);
      MultAAt(gshape, pelmat);
      gshape.GradToDiv(divshape);

      M = mu->Eval(Trans, ip);
      if (lambda)
      {
        L = lambda->Eval(Trans, ip);
      }
      else
      {
        L = q_lambda * M;
        M = q_mu * M;
      }

      if (L != 0.0)
      {
        AddMult_a_VVt(L * w, divshape, elmat);
      }

      if (M != 0.0)
      {
        for (int d = 0; d < dim; d++)
        {
          for (int k = 0; k < dof; k++)
            for (int l = 0; l < dof; l++)
            {
              elmat(dof*d + k, dof*d + l) += (M * w) * pelmat(k, l);
            }
        }
        for (int i = 0; i < dim; i++)
          for (int j = 0; j < dim; j++)
          {
            for (int k = 0; k < dof; k++)
              for (int l = 0; l < dof; l++)
                elmat(dof*i + k, dof*j + l) +=
                (M * w) * gshape(k, j) * gshape(l, i);
            // + (L * w) * gshape(k, i) * gshape(l, j)
          }
      }
    }
  }
};


Mesh* MeshFromPly(std::string filename);
void SamplePly(FiniteElementSpace* fespace_ply, Mesh* plymesh, Mesh* mesh, int dim, GridFunction &x, GridFunction &x_ply);

int main(int argc, char *argv[])
{
  Mesh* plymesh  = MeshFromPly("block.ply");
  Vector bbmax, bbmin;
  plymesh->GetBoundingBox(bbmin,bbmax);
  //shift mesh to 0,0,0
  for (int i = 0; i < plymesh->GetNV(); i++) {
    auto vertex = plymesh->GetVertex(i);
    vertex[0] -= bbmin[0];
    vertex[1] -= bbmin[1];
    vertex[2] -= bbmin[2];
  }

  bbmin = 0.0;
  bbmax -= bbmin;
  cout << "bbox [0,0,0] - [" << bbmax[0] << ", " << bbmax[1] << ", " << bbmax[2] << "\n";

  int order = 1;
  bool static_cond = false;
  bool visualization = 1;

  int dim = 3;
  Mesh *mesh = new Mesh("bridge_moments.vtk");//new Mesh(10, 10, 10, mfem::Element::HEXAHEDRON, 0, bbmax[0], bbmax[1], bbmax[2]);

  cout << "total number of elements: " << mesh->GetNE() << "\n";
  cout << "total number of boundary elements: " << mesh->GetNBE() << "\n";
  auto att = mesh->GetElement(1)->GetAttribute();
  cout << "total number of element attributes " << att << "\n";

  mesh->FinalizeTopology();
  mesh->Finalize();

  int fixed_bdratt = 1, force_bdratt = 6;
  {
    ofstream mesh_ofs("block.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
  }

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 1000
  //    elements.
  {
    int ref_levels = 4;
        //(int)floor(log(5000./plymesh->GetNE())/log(2.)/dim);
     for (int l = 0; l < ref_levels; l++)
     {
        plymesh->UniformRefinement();
     }
  }

  FiniteElementCollection *fec;
  FiniteElementSpace *fespace;
  fec = new H1_FECollection(order, dim);
  fespace = new FiniteElementSpace(mesh, fec, dim);

  FiniteElementCollection *fec_ply;
  FiniteElementSpace *fespace_ply;
  fec_ply = new H1_FECollection(order, dim);
  fespace_ply = new FiniteElementSpace(plymesh, fec_ply, dim);

  //initialize the global parameters for moment fitting
  QuadratureGenerator::InitializeNormalizedQuadratures(5);

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking only
  //    boundary attribute 1 from the mesh as essential and converting it to a
  //    list of true dofs.
  Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
  {
    ess_bdr = 0;
    ess_bdr[fixed_bdratt - 1] = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  VectorArrayCoefficient f(dim);
  {
    for (int i = 0; i < dim - 1; i++)
    {
      f.Set(i, new ConstantCoefficient(0.0));
    }
    {
      Vector pull_force(mesh->bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(force_bdratt - 1) = 1.0;
      f.Set(dim - 1, new PWConstCoefficient(pull_force));
    }
  }

  LinearForm *b = new LinearForm(fespace);
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
  b->Assemble();

  GridFunction x(fespace);
  x = 0.0;

  ConstantCoefficient lambda_func(100.0);
  ConstantCoefficient mu_func(30.0);

  BilinearForm *a = new BilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegratorCustom(lambda_func, mu_func));

  cout << "matrix ... " << flush;
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  SparseMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
  cout << "done." << endl;
  cout << "Size of linear system: " << A.Height() << endl;

  GSSmoother M(A);
  PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
  a->RecoverFEMSolution(X, *b, x);
  cout << "Max/Min x Value: " << x.Max() << "/" << x.Min() << "\n";

  GridFunction x_ply(fespace_ply);
  x_ply = 0.0;
  SamplePly(fespace_ply, plymesh, mesh, dim, x, x_ply);

  if (!mesh->NURBSext)
  {
    mesh->SetNodalFESpace(fespace);
  }

  {
    ofstream mesh_ofs("ex1orig.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);

    GridFunction *nodes = mesh->GetNodes();
    //*nodes += x;
    //x *= -1;

    ofstream vtk_ofs("ex1sol.vtk");
    mesh->PrintVTK(vtk_ofs, 1);
    x.SaveVTK(vtk_ofs, "displacement", 1);

    ofstream vtk_ofs_ply("ex1sol_ply.vtk");
    plymesh->PrintVTK(vtk_ofs_ply, 1);
    x_ply.SaveVTK(vtk_ofs_ply, "displacement", 1);

  //compute stress

    // A. Define a finite element space for post-processing the solution. We
    //    use a discontinuous space of the same order as the solution. L2
    H1_FECollection stress_fec(order, dim);
    FiniteElementSpace stress_fespace(plymesh, &stress_fec);
    GridFunction stress_field(&stress_fespace);


    // B. Project the post-processing coefficient defined above to the
    //    'pp_field' GridFunction.
    MyCoefficient stress_coeff(x_ply, lambda_func, mu_func);
    stress_field.ProjectCoefficient(stress_coeff);
    //stress_field.SaveVTK(vtk_ofs_ply, "stress", 1);
    stress_field.SaveVTK(vtk_ofs_ply, "stress", 1);
  }

  {
    delete a;
    delete b;
    if (fec)
    {
      delete fespace;
      delete fec;
    }
    delete mesh;
    delete plymesh;
    return 0;
  }
}

void GetMFIntegrationRule(IntegrationRule& intrule, MomentCell cell) {
  unsigned order = cell.order;
  unsigned moment_size = (unsigned)pow(order + 1, 3);
  vec3 origin, cell_size;
  Vector moment_vector(moment_size);
  moment_vector = 0.0;

  //assert(moment_size == cell.moments.size());
  if (cell.is_boundary) {
    for (unsigned j = 0; j < moment_size; j++) {
      moment_vector(j) = cell.moments[j];
    }
  }

  //compute quadrature points
  QuadratureGenerator quadrature(moment_vector, cell.origin, cell.scale_factor, order, cell.is_boundary);

  for (long k = 0; k < quadrature.m_quad_weights.Size(); k++) {
    IntegrationPoint ip;
    ip.Set(quadrature.m_quad_points[k][0], quadrature.m_quad_points[k][1], quadrature.m_quad_points[k][2], quadrature.m_quad_weights[k]);
    intrule.Append(ip);
  }
}

void ReadMoments(string moments_filename) {
  std::string filename = moments_filename;
  std::ifstream file(filename);

  if (!file.good()) {
    throw std::runtime_error("Moment file not accessible");
  }
  json moment_json;
  file >> moment_json;
  {
    std::vector<double> const bounds = moment_json["bounding_box"];
    std::array<int, 3> const nbins = moment_json["n_bins"];
    double const cell_length = moment_json["cell_length"];

    for (json const& instance : moment_json["instances"]) {
      std::string instance_id = instance["instance_id"];
      cells.resize(instance["bins"].size());

      for (size_t bin_index = 0; bin_index < instance["bins"].size(); bin_index++) {
        auto const& bin = instance["bins"][bin_index];
        MomentCell& cell = cells[bin_index];

        IndexTriplet index = { (unsigned long)bin["i"], (unsigned long)bin["j"], (unsigned long)bin["k"] };
        cell.index = index;

        cell.order = bin["order"];
        if (cell.order > 0) {
          cell.is_boundary = true;
          cell.scale_factor = { 1.0, 1.0, 1.0 };
          cell.origin = { 0.5, 0.5, 0.5 }; //bin["origin"];
          cell.moments = bin["moment_vector"].get<std::vector<double>>();
        }
        else {
          //interior cell
          cell.order = 1;
          cell.scale_factor = { 1.0, 1.0, 1.0 };
          cell.origin = { 0.5, 0.5, 0.5 };
        }
      }
    }
  }
}

void SamplePly(FiniteElementSpace* fespace_ply, Mesh* plymesh, Mesh* mesh, int dim, GridFunction &x, GridFunction &x_ply ) {
  //sample ply mesh
  double * ply_data = x_ply.GetData();
  int ply_data_size = x_ply.Size();
  cout << "Size " << ply_data_size << endl;
  int index = 0;
  for (int i = 0; i < plymesh->GetNV(); i++) {
    Vector point(plymesh->GetVertex(i), 3);
    DenseMatrix points(dim, 1);
    points.SetCol(0, point);
    Array<int> elem_ids;
    Array<IntegrationPoint> ips;
    mesh->FindPoints(points, elem_ids, ips);
    /*cout << "Point: " << point[0] << " " << point[1] << " " << point[2] << " ";
    cout << "Element Id: " << elem_ids[0] << " size " << elem_ids.Size() << " relative point " << ips[0].x << ", " << ips[0].y << ", " << ips[0].z << " ";*/
    //cout << x.GetValue(elem_ids[0], ips[0], 0) << ", " << x.GetValue(elem_ids[0], ips[0]) <<", " << x.GetValue(elem_ids[0], ips[0],2)  <<endl;

    Array<int> dof(1), vdof(3);
    dof = -1;
    vdof = -1;
    fespace_ply->GetVertexDofs(i, dof);
    /*cout << "dof: " << dof[0] << ", vdof: " << fespace_ply->DofToVDof(dof[0], 0) << ", " << fespace_ply->DofToVDof(dof[0], 1) << ", " << fespace_ply->DofToVDof(dof[0], 2) << " ";*/

    /*cout << "Displacement value [" << x.GetValue(elem_ids[0], ips[0], 0) << ", " << x.GetValue(elem_ids[0], ips[0]) << ", " << x.GetValue(elem_ids[0], ips[0], 2) << "]\n";*/
    ply_data[fespace_ply->DofToVDof(dof[0], 0)] = x.GetValue(elem_ids[0], ips[0], 1);
    ply_data[fespace_ply->DofToVDof(dof[0], 1)] = x.GetValue(elem_ids[0], ips[0], 2);
    ply_data[fespace_ply->DofToVDof(dof[0], 2)] = x.GetValue(elem_ids[0], ips[0], 3);
  }

}


Mesh* MeshFromPly(std::string filename) {
  /*try
  {
  std::ifstream ss(filename);

  if (!ss.good()) {
  throw std::runtime_error("File not accessible");
  }

  tinyply::PlyFile file(ss);

  vector<uint32_t> faces;
  unsigned long faceCount = (unsigned long)file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);
  if (faceCount == 0) {
  faceCount = (unsigned long)file.request_properties_from_element("face", { "vertex_index" }, faces, 3);
  }

  vector<double> verts;
  file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);
  file.read(ss);

  Mesh* plymesh = new Mesh(2, verts.size(), faces.size());
  unsigned long i = 0;
  while (i < verts.size()) {
  vector<double> vertex = { (double)verts[i++], (double)verts[i++], (double)verts[i++] };
  plymesh->AddVertex(vertex.data());
  }

  i = 0;
  while (i < faces.size()) {
  vector<int> v_idx = { (int)faces[i++] , (int)faces[i++] , (int)faces[i++] };
  plymesh->AddBdrTriangle(v_idx.data());
  }

  plymesh->FinalizeTopology();
  plymesh->Finalize();
  return plymesh;
  }
  catch (const std::runtime_error& error)*/
  {
    std::ifstream ss(filename);

    if (!ss.good()) {
      throw std::runtime_error("File not accessible");
    }

    tinyply::PlyFile file(ss);

    std::vector<uint32_t> faces;
    unsigned long faceCount = (unsigned long)file.request_properties_from_element("face", { "vertex_indices" }, faces, 3);
    if (faceCount == 0) {
      faceCount = (unsigned long)file.request_properties_from_element("face", { "vertex_index" }, faces, 3);
    }
    std::vector<float> verts;
    file.request_properties_from_element("vertex", { "x", "y", "z" }, verts);
    file.read(ss);

    Mesh* plymesh = new Mesh(2, (int)verts.size() / 3, (int)faces.size() / 3, 0, 3);
    unsigned long i = 0;
    while (i < verts.size()) {
      vector<double> vertex = { (double)verts[i++], (double)verts[i++], (double)verts[i++] };
      plymesh->AddVertex(vertex.data());
    }

    i = 0;
    while (i < faces.size()) {
      vector<int> v_idx = { (int)faces[i++] , (int)faces[i++] , (int)faces[i++] };
      plymesh->AddTriangle(v_idx.data());
    }
    plymesh->FinalizeTopology();
    plymesh->Finalize();
    ofstream mesh_ofs("blockply.vtk");
    mesh_ofs.precision(8);
    plymesh->PrintVTK(mesh_ofs);
    return plymesh;
  }
}


