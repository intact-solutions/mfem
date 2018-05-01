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

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
  int order = 1;
  bool static_cond = false;
  bool visualization = 1;

  Mesh *mesh = new Mesh(3, 8, 1, 6);
  int dim = 3;

  // Creating a one-element mesh:
  vector<double> pt = { 0.0,0.0,0.0 };
  mesh->AddVertex(pt.data());
  pt = { 1.0 , 0.0, 0.0 };
  mesh->AddVertex(pt.data());
  pt = { 1.0 , 1.0, 0.0 };
  mesh->AddVertex(pt.data());
  pt = { 0.0 , 1.0, 0.0 };
  mesh->AddVertex(pt.data());
  pt = { 0.0 , 0.0, 1.0 };
  mesh->AddVertex(pt.data());
  pt = { 1.0 , 0.0, 1.0 };
  mesh->AddVertex(pt.data());
  pt = { 1.0 , 1.0, 1.0 };
  mesh->AddVertex(pt.data());
  pt = { 0.0 , 1.0, 1.0 };
  mesh->AddVertex(pt.data());

  vector<int> els = { 0,1,2,3,4,5,6,7 };
  mesh->AddHex(els.data());

  els = { 0,3,2,1 };
  mesh->AddBdrQuad(els.data());
  els = { 0,1,5,4 };
  mesh->AddBdrQuad(els.data());
  els = { 3,0,4,7 };
  mesh->AddBdrQuad(els.data());
  els = { 1,2,6,5 };
  mesh->AddBdrQuad(els.data());
  els = { 6,2,3,7 };
  mesh->AddBdrQuad(els.data());
  els = { 4,5,6,7 };
  mesh->AddBdrQuad(els.data());

  mesh->FinalizeTopology();
  mesh->Finalize();

  {
    ofstream mesh_ofs("block.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
  }

  FiniteElementCollection *fec;
  FiniteElementSpace *fespace;
  fec = new H1_FECollection(order, dim);
  fespace = new FiniteElementSpace(mesh, fec, dim);

  // manual definition of boundary conditions: nodes 0,1,2,3 are clamped (dofs set to zero)
  Array<int> ess_tdof_list;
  ess_tdof_list.SetSize(12);
  ess_tdof_list[0] = 2;
  ess_tdof_list[1] = 3;
  ess_tdof_list[2] = 6;
  ess_tdof_list[3] = 7;
  ess_tdof_list[4] = 10;
  ess_tdof_list[5] = 11;
  ess_tdof_list[6] = 14;
  ess_tdof_list[7] = 15;
  ess_tdof_list[8] = 18;
  ess_tdof_list[9] = 19;
  ess_tdof_list[10] = 22;
  ess_tdof_list[11] = 23;

  // manual definition of rhs grid function: force along X on face composed of nodes 0-1-4-5
  mfem::GridFunction my_vec(fespace);
  my_vec[0] = 1.0;
  my_vec[1] = 1.0;
  my_vec[4] = 1.0;
  my_vec[5] = 1.0;

  // Then we build a Coefficient from this vec
  VectorGridFunctionCoefficient f(&my_vec);

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
    GridFunction *nodes = mesh->GetNodes();
    *nodes += x;
    x *= -1;
    ofstream mesh_ofs("displaced.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x.Save(sol_ofs);
  }

  delete a;
  delete b;
  if (fec)
  {
    delete fespace;
    delete fec;
  }
  delete mesh;

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
