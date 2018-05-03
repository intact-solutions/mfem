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
#include"../IntactMods/tinyply/source/tinyply.h"

using namespace std;
using namespace mfem;


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

    Mesh* plymesh = new Mesh(2, (int)verts.size()/3, (int)faces.size()/3,0,3);
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
  Mesh *mesh = new Mesh(10, 10, 10, mfem::Element::HEXAHEDRON, 0, bbmax[0], bbmax[1], bbmax[2]);


  cout << "total number of elements: " << mesh->GetNE() << "\n";
  cout << "total number of boundary elements: " << mesh->GetNBE() << "\n";

  int fixed_bdratt = 2, force_bdratt = 3;

  //mesh->GetBdrElement(0)->SetAttribute(fixed_bdratt);
  //mesh->GetBdrElement(5)->SetAttribute(force_bdratt);

  mesh->FinalizeTopology();
  mesh->Finalize();

  for (int i = 0; i < mesh->GetNBE(); i++)
  {
    auto bdrface = mesh->GetBdrElement(i);
    Array<int> vertex_idx;
    bdrface->GetVertices(vertex_idx);

    bool res_face = true, load_face = true;
    //even if one vertex is on plane to z == 0 or
    for (int j = 0; j < vertex_idx.Size(); j++) {
      double coord = mesh->GetVertex(vertex_idx[0])[2];
      if (coord > .01)
        res_face = false;
      else if (coord < 29.99)
        load_face = false;
    }
    if(res_face)
      bdrface->SetAttribute(fixed_bdratt);
    if(load_face)
      bdrface->SetAttribute(force_bdratt);
  }

  {
    ofstream mesh_ofs("block.vtk");
    mesh_ofs.precision(8);
    mesh->PrintVTK(mesh_ofs);
  }

  //// 4. Refine the mesh to increase the resolution. In this example we do
  ////    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  ////    largest number that gives a final mesh with no more than 1000
  ////    elements.
  //{
  //   int ref_levels =
  //      (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
  //   for (int l = 0; l < ref_levels; l++)
  //   {
  //      mesh->UniformRefinement();
  //   }
  //}

  FiniteElementCollection *fec;
  FiniteElementSpace *fespace;
  fec = new H1_FECollection(order, dim);
  fespace = new FiniteElementSpace(mesh, fec, dim);

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking only
  //    boundary attribute 1 from the mesh as essential and converting it to a
  //    list of true dofs.
  Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[fixed_bdratt-1] = 1;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

  VectorArrayCoefficient f(dim);
  for (int i = 0; i < dim - 1; i++)
  {
    f.Set(i, new ConstantCoefficient(0.0));
  }
  {
    Vector pull_force(mesh->bdr_attributes.Max());
    pull_force = 0.0;
    pull_force(force_bdratt-1) = 1.0;
    f.Set(dim - 1, new PWConstCoefficient(pull_force));
  }

  LinearForm *b = new LinearForm(fespace);
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
  b->Assemble();

  GridFunction x(fespace);
  x = 0.0;

  ConstantCoefficient lambda_func(100.0);
  ConstantCoefficient mu_func(30.0);

  BilinearForm *a = new BilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

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

  if (!mesh->NURBSext)
  {
    mesh->SetNodalFESpace(fespace);
  }

  {
    ofstream mesh_ofs("ex1orig.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);

    GridFunction *nodes = mesh->GetNodes();
    *nodes += x;
    x *= -1;

    ofstream vtk_ofs("ex1sol.vtk");
    mesh->PrintVTK(vtk_ofs, 1);
    x.SaveVTK(vtk_ofs, "displacement", 1);
  }

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

int main2(int argc, char *argv[])
{
  // 1. Parse command-line options.
  const char *mesh_file = "../data/fichera-q2.vtk";
  int order = 1;
  bool static_cond = false;
  bool visualization = 1;

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
    "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
    "Finite element order (polynomial degree) or -1 for"
    " isoparametric space.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
    "--no-static-condensation", "Enable static condensation.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
    "--no-visualization",
    "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good())
  {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
  //    the same code.
  Mesh *mesh = new Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 3. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 50,000
  //    elements.
  {
    int ref_levels =
      (int)floor(log(50000. / mesh->GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++)
    {
      mesh->UniformRefinement();
    }
  }

  // 4. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  if (order > 0)
  {
    fec = new H1_FECollection(order, dim);
  }
  else if (mesh->GetNodes())
  {
    fec = mesh->GetNodes()->OwnFEC();
    cout << "Using isoparametric FEs: " << fec->Name() << endl;
  }
  else
  {
    fec = new H1_FECollection(order = 1, dim);
  }
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
  cout << "Number of finite element unknowns: "
    << fespace->GetTrueVSize() << endl;

  // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
  {
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  // 6. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  LinearForm *b = new LinearForm(fespace);
  ConstantCoefficient one(1.0);
  b->AddDomainIntegrator(new DomainLFIntegrator(one));
  b->Assemble();

  // 7. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  GridFunction x(fespace);
  x = 0.0;

  // 8. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    domain integrator.
  BilinearForm *a = new BilinearForm(fespace);
  a->AddDomainIntegrator(new DiffusionIntegrator(one));

  // 9. Assemble the bilinear form and the corresponding linear system,
  //    applying any necessary transformations such as: eliminating boundary
  //    conditions, applying conforming constraints for non-conforming AMR,
  //    static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  SparseMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

  cout << "Size of linear system: " << A.Height() << endl;

#ifndef MFEM_USE_SUITESPARSE
  // 10. Define a simple symmetric Gauss-Seidel preconditioner and use it to
  //     solve the system A X = B with PCG.
  GSSmoother M(A);
  PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
  // 10. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(A);
  umf_solver.Mult(B, X);
#endif

  // 11. Recover the solution as a finite element grid function.
  a->RecoverFEMSolution(X, *b, x);

  // 12. Save the refined mesh and the solution. This output can be viewed later
  //     using GLVis: "glvis -m refined.mesh -g sol.gf".
  ofstream mesh_ofs("refined.mesh");
  mesh_ofs.precision(8);
  mesh->Print(mesh_ofs);
  ofstream sol_ofs("sol.gf");
  sol_ofs.precision(8);

  GridFunction *nodes = mesh->GetNodes();
  *nodes += x;
  x *= -1;

  x.Save(sol_ofs);

  // 13. Send the solution by socket to a GLVis server.
  if (visualization)
  {
    char vishost[] = "localhost";
    int  visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << *mesh << x << flush;
  }

  // 14. Free the used memory.
  delete a;
  delete b;
  delete fespace;
  if (order > 0) { delete fec; }
  delete mesh;

  return 0;
}
