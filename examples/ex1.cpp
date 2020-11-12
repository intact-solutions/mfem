//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/star-mixed.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/fichera-mixed.mesh
//               ex1 -m ../data/toroid-wedge.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/star-mixed-p2.mesh -o 2
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/fichera-mixed-p2.mesh -o 2
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               ex1 -pa -d cuda
//               ex1 -pa -d raja-cuda
//               ex1 -pa -d occa-cuda
//               ex1 -pa -d raja-omp
//               ex1 -pa -d occa-omp
//               ex1 -pa -d ceed-cpu
//               ex1 -pa -d ceed-cuda
//               ex1 -m ../data/beam-hex.mesh -pa -d cuda
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

using namespace std;
using namespace mfem;

GridFunction DarcyFlowSolver(Mesh* mesh, int order, bool pa, bool static_cond);
void velocity_function(const Vector& x, Vector& v) {
  int dim = x.Size();

  // map to the reference [-1,1] domain
  v.SetSize(dim);
  v[0] = 100; 
  v[1] = 0.0;
  v[2] = 0.0;
};

/// alpha (q . grad u, v)
class ConvectionIntegratorStabilized : public BilinearFormIntegrator
{
protected:
  VectorCoefficient* Q;
  double alpha;

private:
#ifndef MFEM_THREAD_SAFE
  DenseMatrix dshape, adjJ, Q_ir;
  Vector shape, vec2, BdFidxT;
#endif

public:
  ConvectionIntegratorStabilized(VectorCoefficient& q, double a = 1.0)
    : Q(&q) {
    alpha = a;
  }
  virtual void AssembleElementMatrix( const FiniteElement& el, ElementTransformation& Trans, DenseMatrix& elmat)
  {
    int nd = el.GetDof();
    int dim = el.GetDim();

#ifdef MFEM_THREAD_SAFE
    DenseMatrix dshape, adjJ, Q_ir;
    Vector shape, vec2, BdFidxT;
#endif
    elmat.SetSize(nd);
    dshape.SetSize(nd, dim);
    adjJ.SetSize(dim);
    shape.SetSize(nd);
    vec2.SetSize(dim);
    BdFidxT.SetSize(nd);

    Vector vec1;

    const IntegrationRule* ir = IntRule;
    if (ir == NULL)
    {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
    }

    Q->Eval(Q_ir, Trans, *ir);
    

    elmat = 0.0;
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);     

      vec1 *= alpha * ip.weight * -1;      

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      double he = std::pow(Trans.Jacobian().Det(), 1/3);
            
      auto vec1_norm2 = vec1.Norml2();
      DenseMatrix normalized_vec1(dim, 1);
      if (vec1_norm2 > 1e-8) {
        for (int j = 0; j < dim; j++)
          normalized_vec1(j, 0) = 0.5* he* vec1[j]/vec1_norm2;
      }
      DenseMatrix stablizer;
      stablizer.SetSize(nd, 1);
      Mult(dshape, normalized_vec1, stablizer);
      Vector shape_mod(nd);
      for (int k = 0; k < nd; k++) {
        shape_mod[k] = shape[k] + stablizer(k, 0);
      }

      AddMultVWt(shape_mod, BdFidxT, elmat);
    }
  };
};


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
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

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   auto pressure_gf = DarcyFlowSolver(mesh, order, pa, static_cond);   

   // 5. Define a finite element space on the mesh. Here we use continuous
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

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking all
   //    the boundary attributes from the mesh as essential (Dirichlet) and
   //    converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[0] = 1;      
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   cout << "\nessential dofs: "; ess_tdof_list.Print();
   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   Vector heat(mesh->bdr_attributes.Max());
   heat = 0.0;
   heat(1) = -1.0;
   PWConstCoefficient q(heat);

   LinearForm* b = new LinearForm(fespace);
   b->AddBoundaryIntegrator(new BoundaryLFIntegrator(q));
   cout << "r.h.s. ... " << flush;
   b->Assemble();

   cout << "\nRHS: "; b->Print();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ConstantCoefficient k(1.0);
   BilinearForm *a = new BilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator(k));
      
   //add the convection term
   GradientGridFunctionCoefficient velocity_new(&pressure_gf);

   VectorFunctionCoefficient velocity(dim, velocity_function);
   a->AddDomainIntegrator(new ConvectionIntegratorStabilized(velocity_new, 1));
  
   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();
   a->Finalize();
 
   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;
   cout << "\nLHS: ";  
   for (int i = 0; i < a->Size(); i++) {
     cout << a->Elem(i, i) << ", ";
   }  
   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
          
      GSSmoother M((SparseMatrix&)(*A));
      GMRES((SparseMatrix&)(*A), M, *b, x, 1, 200, 10, 1e-12, 0.0);
#else
      // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
      UMFPackSolver umf_solver;
      umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
      umf_solver.SetOperator(*A);
      umf_solver.Mult(B, X);
#endif
   }
   else // Jacobi preconditioning in partial assembly mode
   {
      OperatorJacobiSmoother M(*a, ess_tdof_list);
      PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
   }

   // 12. Recover the solution as a finite element grid function.
   a->RecoverFEMSolution(X, *b, x);

   // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream vtk_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\conjugate_sol.vtk");
   vtk_ofs.precision(8);
   mesh->PrintVTK(vtk_ofs, 1, 0);
   x.SaveVTK(vtk_ofs, "Temp", 1);

   // 15. Free the used memory.
   delete a;
   delete b;  
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}

GridFunction DarcyFlowSolver(Mesh* mesh, int order, bool pa, bool static_cond) {

  int dim = mesh->Dimension();

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order. If order < 1, we
  //    instead use an isoparametric/isogeometric space.
  FiniteElementCollection* fec;
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
  FiniteElementSpace* fespace = new FiniteElementSpace(mesh, fec);
  cout << "Number of finite element unknowns: "
    << fespace->GetTrueVSize() << endl;

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking all
  //    the boundary attributes from the mesh as essential (Dirichlet) and
  //    converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (mesh->bdr_attributes.Size())
  {
    Array<int> ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[1] = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }
  cout << "\nessential dofs: "; ess_tdof_list.Print();
  // 7. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
  //    the basis functions in the finite element fespace.
  Vector vel(mesh->bdr_attributes.Max());
  vel = 0.0;
  vel(0) = 10.0;
  PWConstCoefficient v(vel);

  LinearForm* b = new LinearForm(fespace);
  b->AddBoundaryIntegrator(new BoundaryLFIntegrator(v));
  cout << "r.h.s. ... " << flush;
  b->Assemble();

  cout << "\nRHS: "; b->Print();

  // 8. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  GridFunction x(fespace);
  x = 0.0;

  // 9. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
  //    domain integrator.
  ConstantCoefficient perm(1.0);
  BilinearForm* a = new BilinearForm(fespace);
  if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
  a->AddDomainIntegrator(new DiffusionIntegrator(perm));

  // 10. Assemble the bilinear form and the corresponding linear system,
  //     applying any necessary transformations such as: eliminating boundary
  //     conditions, applying conforming constraints for non-conforming AMR,
  //     static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();
  a->Finalize();

  OperatorPtr A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

  cout << "Size of linear system: " << A->Height() << endl;
  cout << "\nLHS: ";
  for (int i = 0; i < a->Size(); i++) {
    cout << a->Elem(i, i) << ", ";
  }
  // 11. Solve the linear system A X = B.
  if (!pa)
  {
#ifndef MFEM_USE_SUITESPARSE
    // Use a simple symmetric Gauss-Seidel preconditioner with PCG.

    GSSmoother M((SparseMatrix&)(*A));
    GMRES((SparseMatrix&)(*A), M, *b, x, 1, 200, 10, 1e-12, 0.0);
#else
    // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
    UMFPackSolver umf_solver;
    umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
    umf_solver.SetOperator(*A);
    umf_solver.Mult(B, X);
#endif
  }
  else // Jacobi preconditioning in partial assembly mode
  {
    OperatorJacobiSmoother M(*a, ess_tdof_list);
    PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
  }

  // 12. Recover the solution as a finite element grid function.
  a->RecoverFEMSolution(X, *b, x);

  // 13. Save the refined mesh and the solution. This output can be viewed later
  //     using GLVis: "glvis -m refined.mesh -g sol.gf".
  ofstream vtk_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\conjugate_pressure.vtk");
  vtk_ofs.precision(8);
  mesh->PrintVTK(vtk_ofs, 1, 0);
  x.SaveVTK(vtk_ofs, "Pressure", 1);

  return x;
};
