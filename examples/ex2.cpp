//                                MFEM Example 2
//
// Compile with: make ex2
//
// Sample runs:  ex2 -m ../data/beam-tri.mesh
//               ex2 -m ../data/beam-quad.mesh
//               ex2 -m ../data/beam-tet.mesh
//               ex2 -m ../data/beam-hex.mesh
//               ex2 -m ../data/beam-wedge.mesh
//               ex2 -m ../data/beam-quad.mesh -o 3 -sc
//               ex2 -m ../data/beam-quad-nurbs.mesh
//               ex2 -m ../data/beam-hex-nurbs.mesh
//
// Description:  This example code solves a simple linear elasticity problem
//               describing a multi-material cantilever beam.
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise. The geometry of the domain is assumed to be as
//               follows:
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               The example demonstrates the use of high-order and NURBS vector
//               finite element spaces with the linear elasticity bilinear form,
//               meshes with curved elements, and the definition of piece-wise
//               constant and vector coefficient objects. Static condensation is
//               also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/** Incremental hyperelastic integrator for any given HyperelasticModel.

    Represents @f$ \int W(Jpt) dx @f$ over a target zone, where W is the
    @a model's strain energy density function, and Jpt is the Jacobian of the
    target->physical coordinates transformation. The target configuration is
    given by the current mesh at the time of the evaluation of the integrator.
*/
class IncrementalHyperelasticIntegrator : public mfem::NonlinearFormIntegrator {
private:
  mfem::HyperelasticModel* model;

  //   Jrt: the Jacobian of the target-to-reference-element transformation.
  //   Jpr: the Jacobian of the reference-to-physical-element transformation.
  //   Jpt: the Jacobian of the target-to-physical-element transformation.
  //     P: represents dW_d(Jtp) (dim x dim).
  //   DSh: gradients of reference shape functions (dof x dim).
  //    DS: gradients of the shape functions in the target (stress-free)
  //        configuration (dof x dim).
  // PMatI: coordinates of the deformed configuration (dof x dim).
  // PMatO: reshaped view into the local element contribution to the operator
  //        output - the result of AssembleElementVector() (dof x dim).
  mfem::DenseMatrix DSh, DS, Jrt, Jpr, Jpt, P, PMatI, PMatO;

public:
  /** @param[in] m  HyperelasticModel that will be integrated. */
  IncrementalHyperelasticIntegrator(mfem::HyperelasticModel* m) : model(m) {}

  /** @brief Computes the integral of W(Jacobian(Trt)) over a target zone
      @param[in] el     Type of FiniteElement.
      @param[in] Ttr    Represents ref->target coordinates transformation.
      @param[in] elfun  Physical coordinates of the zone. */
  virtual double GetElementEnergy(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
    const mfem::Vector& elfun);

  virtual void AssembleElementVector(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
    const mfem::Vector& elfun, mfem::Vector& elvect);

  virtual void AssembleElementGrad(const mfem::FiniteElement& el, mfem::ElementTransformation& Ttr,
    const mfem::Vector& elfun, mfem::DenseMatrix& elmat);
};

class HyperelasticTractionIntegrator : public mfem::NonlinearFormIntegrator {
private:
  mfem::VectorCoefficient& function;
  mutable mfem::DenseMatrix DSh_u, DS_u, J0i, F, Finv, FinvT, PMatI_u;
  mutable mfem::Vector      shape, nor, fnor, Sh_p, Sh_u;

public:
  HyperelasticTractionIntegrator(mfem::VectorCoefficient& f) : function(f) {}

  virtual void AssembleFaceVector(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
    mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun, mfem::Vector& elvec);

  virtual void AssembleFaceGrad(const mfem::FiniteElement& el1, const mfem::FiniteElement& el2,
    mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
    mfem::DenseMatrix& elmat);

  virtual ~HyperelasticTractionIntegrator() {}
};



/// The abstract MFEM operator for a quasi-static solve
class NonlinearSolidQuasiStaticOperator : public mfem::Operator {
protected:
  /// The nonlinear form
  std::shared_ptr<mfem::NonlinearForm> m_H_form;

  /// The linearized jacobian at the current state
  mutable std::unique_ptr<mfem::Operator> m_Jacobian;

public:
  /// The constructor
  NonlinearSolidQuasiStaticOperator(std::shared_ptr<mfem::NonlinearForm> H_form);

  /// Required to use the native newton solver
  mfem::Operator& GetGradient(const mfem::Vector& x) const;

  /// Required for residual calculations
  void Mult(const mfem::Vector& k, mfem::Vector& y) const;

  /// The destructor
  virtual ~NonlinearSolidQuasiStaticOperator();
};

void InitialDeformation(const Vector& x, Vector& y);
GridFunction elasticity_main(Mesh* mesh, double lambda, double mu, double force);
int main(int argc, char *argv[])
{  
   // 1. Parse command-line options.
   const char *mesh_file = "../data/beam-hex.mesh";
   int order = 1;
   bool static_cond = false;
   bool visualization = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
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
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   Mesh* mesh = new Mesh(mesh_file, 1, 1);   
   Mesh* mesh2 = new Mesh(mesh_file, 1, 1);
   //Mesh* mesh = new Mesh(20, 10.0);
   //Mesh* mesh2 = new Mesh(20, 10.0); 
  
   int dim = mesh->Dimension();

   //if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
   //{
   //   cerr << "\nInput mesh should have at least two materials and "
   //        << "two boundary attributes! (See schematic in ex2.cpp)\n"
   //        << endl;
   //   return 3;
   //}

   // 3. Select the order of the finite element discretization space. For NURBS
   //    meshes, we increase the order by degree elevation.
   if (mesh->NURBSext)
   {
      mesh->DegreeElevate(order, order);
   }

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 5,000
   //    elements.
   {
     int ref_levels = 1;
         //(int)floor(log(5000./mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
         mesh2->UniformRefinement();
      }
   }  

   // 5. Define a finite element space on the mesh. Here we use vector finite
   //    elements, i.e. dim copies of a scalar finite element space. The vector
   //    dimension is specified by the last argument of the FiniteElementSpace
   //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
   //    associated with the mesh nodes.
   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;
   if (mesh->NURBSext)
   {
      fec = NULL;
      fespace = mesh->GetNodes()->FESpace();
   }
   else
   {
      fec = new H1_FECollection(order, dim);
      fespace = new FiniteElementSpace(mesh, fec, dim);
   }

   int fe_size = fespace->GetTrueVSize();
   cout << "Number of finite element unknowns: " << fe_size
        << endl << "Assembling: " << flush;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   cout << "\nessential dofs: ";
   //ess_tdof_list.Print();

    // define the traction vector
   Array<int> trac_bdr(mesh->bdr_attributes.Max());
   trac_bdr = 0;
   trac_bdr[1] = 1;
     

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the linear elasticity integrator with piece-wise
   //    constants coefficient lambda and mu.   
   //ConstantCoefficient lambda(1);
   //ConstantCoefficient mu(1);
   double mu = 0.25, K = 150.0;  
   //double mu = 1.0, K = 1.67;

   Vector rhs;
   rhs = 0.0;
   
   GridFunction uh(fespace); //solution
   uh = 0.0;

   double force_step = 10.0e-4;
   int num_steps = 50;
   int sampling_steps = 5;
   for (int i = 0; i < num_steps; i++) {

     auto nl_form = std::make_shared<mfem::NonlinearForm>(fespace);
     HyperelasticModel* model = new NeoHookeanModel(mu, K);
     nl_form->AddDomainIntegrator(new IncrementalHyperelasticIntegrator(model));
     nl_form->SetEssentialTrueDofs(ess_tdof_list);
       
     mfem::Vector traction(dim);
     traction = 0.0;
     traction(1) = (i+1) * force_step;
     mfem::VectorConstantCoefficient traction_coeff(traction);
     nl_form->AddBdrFaceIntegrator(new HyperelasticTractionIntegrator(traction_coeff), trac_bdr);

     std::cout << "\n\nSolving for force step: " << i << ": " << (i + 1) * force_step << "\n";

     NonlinearSolidQuasiStaticOperator N_oper(nl_form);

     Solver* J_solver;
     Solver* J_prec = new DSmoother(1);
     MINRESSolver* J_minres = new MINRESSolver;
     J_minres->SetRelTol(1e-6);
     J_minres->SetAbsTol(1e-8);
     J_minres->SetMaxIter(2000);
     J_minres->SetPreconditioner(*J_prec);
     J_solver = J_minres;

     NewtonSolver newton_solver;
     newton_solver.SetRelTol(1e-4);
     newton_solver.SetAbsTol(1e-6);
     newton_solver.SetMaxIter(2000);
     newton_solver.SetSolver(*J_solver);
     newton_solver.SetOperator(N_oper);
     newton_solver.SetPrintLevel(1);
     newton_solver.iterative_mode = true;

     newton_solver.Mult(rhs, uh); //solve the non-linear form with right hand side as rhs and uh has the initial guess (and will eventually store the result)

     //13. Save the refined mesh and the solution.
     if (i % sampling_steps == 0) {
       ofstream mesh_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\hyperelastic_sol_" + std::to_string(i) + ".vtk");
       mesh->PrintVTK(mesh_ofs, 1, 0);
       uh.SaveVTK(mesh_ofs, "u", 1);
     }
   }  

   return 0;
   if (fec)
   {
      delete fespace;
      delete fec;
   }
   delete mesh;

   return 0;
}

void InitialDeformation(const Vector& x, Vector& y)
{
  // Set the initial configuration. Having this different from the reference
  // configuration can help convergence
  y = x;
  y[2] = x[2] + 0.25 * x[2];
}


GridFunction elasticity_main(Mesh* mesh, double lambda, double mu, double force)
{
  // 1. Parse command-line options.
  const char* mesh_file = "../data/beam-quad.mesh";
  int order = 1;
  bool static_cond = false;
  bool visualization = 1;

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral or hexahedral elements with the same code.
  //Mesh* mesh = new Mesh(mesh_file, 1, 1);
  //Mesh* mesh = new Mesh(20, 10.0);
  int dim = mesh->Dimension();

  /*if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
  {
    cerr << "\nInput mesh should have at least two materials and "
      << "two boundary attributes! (See schematic in ex2.cpp)\n"
      << endl;
    return 3;
  }*/

  // 3. Select the order of the finite element discretization space. For NURBS
  //    meshes, we increase the order by degree elevation.
  if (mesh->NURBSext)
  {
    mesh->DegreeElevate(order, order);
  }

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 5,000
  //    elements.
  {
    int ref_levels = 0;
      //(int)floor(log(5000. / mesh->GetNE()) / log(2.) / dim);
    for (int l = 0; l < ref_levels; l++)
    {
      mesh->UniformRefinement();
    }
  }

  // 5. Define a finite element space on the mesh. Here we use vector finite
  //    elements, i.e. dim copies of a scalar finite element space. The vector
  //    dimension is specified by the last argument of the FiniteElementSpace
  //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
  //    associated with the mesh nodes.
  FiniteElementCollection* fec;
  FiniteElementSpace* fespace;
  if (mesh->NURBSext)
  {
    fec = NULL;
    fespace = mesh->GetNodes()->FESpace();
  }
  else
  {
    fec = new H1_FECollection(order, dim);
    fespace = new FiniteElementSpace(mesh, fec, dim);
  }

  cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
    << endl << "Assembling: " << flush;

  // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
  //    In this example, the boundary conditions are defined by marking only
  //    boundary attribute 1 from the mesh as essential and converting it to a
  //    list of true dofs.
  Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[0] = 1;
  fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  //cout << "\nEssential dofs: "; ess_tdof_list.Print();
  // 7. Set up the linear form b(.) which corresponds to the right-hand side of
  //    the FEM linear system. In this case, b_i equals the boundary integral
  //    of f*phi_i where f represents a "pull down" force on the Neumann part
  //    of the boundary and phi_i are the basis functions in the finite element
  //    fespace. The force is defined by the VectorArrayCoefficient object f,
  //    which is a vector of Coefficient objects. The fact that f is non-zero
  //    on boundary attribute 2 is indicated by the use of piece-wise constants
  //    coefficient for its last component.
  VectorArrayCoefficient f(dim);
  for (int i = 0; i < dim - 1; i++)
  {
    f.Set(i, new ConstantCoefficient(0.0));
  }
  {
    Vector pull_force(mesh->bdr_attributes.Max());
    pull_force = 0.0;
    pull_force(1) = force;
    f.Set(1, new PWConstCoefficient(pull_force));
  }

  LinearForm* b = new LinearForm(fespace);
  b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
  cout << "r.h.s. ... " << flush;
  b->Assemble();
  //cout << "rhs: "; b->Print();
  // 8. Define the solution vector x as a finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  GridFunction x(fespace);
  x = 0.0;

  // 9. Set up the bilinear form a(.,.) on the finite element space
  //    corresponding to the linear elasticity integrator with piece-wise
  //    constants coefficient lambda and mu.
  
  ConstantCoefficient lambda_func(lambda);
  ConstantCoefficient mu_func(mu);

  BilinearForm* a = new BilinearForm(fespace);
  a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

  // 10. Assemble the bilinear form and the corresponding linear system,
  //     applying any necessary transformations such as: eliminating boundary
  //     conditions, applying conforming constraints for non-conforming AMR,
  //     static condensation, etc.
  cout << "matrix ... " << flush;
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  SparseMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
  cout << "done." << endl;

  cout << "Size of linear system: " << A.Height() << endl;  
  //cout << "\nLHS: "; A.Print();

#ifndef MFEM_USE_SUITESPARSE
  // 11. Define a simple symmetric Gauss-Seidel preconditioner and use it to
  //     solve the system Ax=b with PCG.
  GSSmoother M(A);
  PCG(A, M, B, X, 1, 500, 1e-8, 0.0);
#else
  // 11. If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
  UMFPackSolver umf_solver;
  umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
  umf_solver.SetOperator(A);
  umf_solver.Mult(B, X);
#endif

  // 12. Recover the solution as a finite element grid function.
  a->RecoverFEMSolution(X, *b, x);

  // 13. For non-NURBS meshes, make the mesh curved based on the finite element
  //     space. This means that we define the mesh elements through a fespace
  //     based transformation of the reference element. This allows us to save
  //     the displaced mesh as a curved mesh when using high-order finite
  //     element displacement field. We assume that the initial mesh (read from
  //     the file) is not higher order curved mesh compared to the chosen FE
  //     space.
  if (!mesh->NURBSext)
  {
    mesh->SetNodalFESpace(fespace);
  }

  // 14. Save the displaced mesh and the inverted solution (which gives the
  //     backward displacements to the original grid). This output can be
  //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
  {
    /*GridFunction* nodes = mesh->GetNodes();
    *nodes += x;
    x *= -1;*/
    /*ofstream mesh_ofs("displaced.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x.Save(sol_ofs);*/

    ofstream mesh_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\elastic_sol.vtk");
    mesh_ofs.precision(8);
    mesh->PrintVTK(mesh_ofs, 1, 0);
    x.SaveVTK(mesh_ofs, "u", 1);
  }

  // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
  //     keys in GLVis to visualize the displacements.
  if (visualization)
  {
    char vishost[] = "localhost";
    int  visport = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << *mesh << x << flush;
  }

  // 16. Free the used memory.
  delete a;
  delete b;
  if (fec)
  {
    delete fespace;
    delete fec;
  }
  delete mesh;

  return x;
}

double IncrementalHyperelasticIntegrator::GetElementEnergy(const mfem::FiniteElement& el,
  mfem::ElementTransformation& Ttr, const mfem::Vector& elfun)
{
  int    dof = el.GetDof(), dim = el.GetDim();
  double energy;

  DSh.SetSize(dof, dim);
  Jrt.SetSize(dim);
  Jpr.SetSize(dim);
  Jpt.SetSize(dim);
  PMatI.UseExternalData(elfun.GetData(), dof, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  energy = 0.0;
  model->SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt);

    el.CalcDShape(ip, DSh);
    MultAtB(PMatI, DSh, Jpr);
    Mult(Jpr, Jrt, Jpt);

    for (int d = 0; d < dim; d++) {
      Jpt(d, d) += 1.0;
    }

    energy += ip.weight * Ttr.Weight() * model->EvalW(Jpt);
  }

  return energy;
}

void IncrementalHyperelasticIntegrator::AssembleElementVector(const mfem::FiniteElement& el,
  mfem::ElementTransformation& Ttr,
  const mfem::Vector& elfun, mfem::Vector& elvect)
{
  int dof = el.GetDof(), dim = el.GetDim();

  DSh.SetSize(dof, dim);
  DS.SetSize(dof, dim);
  Jrt.SetSize(dim);
  Jpt.SetSize(dim);
  P.SetSize(dim);
  PMatI.UseExternalData(elfun.GetData(), dof, dim);
  elvect.SetSize(dof * dim);
  PMatO.UseExternalData(elvect.GetData(), dof, dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elvect = 0.0;
  model->SetTransformation(Ttr);
  
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt);

    el.CalcDShape(ip, DSh);
    Mult(DSh, Jrt, DS);
    MultAtB(PMatI, DS, Jpt);

    for (int d = 0; d < dim; d++) {
      Jpt(d, d) += 1.0;
    }

    auto det = Jpt.Det();
    //std::cout << " " << det;
    if (det < 0 ) {
      //std::cout << "\n-ve Jac;";
      /*std::cout << "\nTtr Jac: ";  Ttr.Jacobian().Print();
      std::cout << "\nJrt: ";  Jrt.Print();
      std::cout << "\nDSh: ";  DSh.Print();
      std::cout << "\nDS: ";  DS.Print();
      std::cout << "\nPMatI: ";  PMatI.Print();
      std::cout << "\nJpt: "; Jpt.Print();*/
    }    
    //Jpt.Print();
    model->EvalP(Jpt, P);

    P *= ip.weight * Ttr.Weight();
    
    AddMultABt(DS, P, PMatO);
  }
}

void IncrementalHyperelasticIntegrator::AssembleElementGrad(const mfem::FiniteElement& el,
  mfem::ElementTransformation& Ttr, const mfem::Vector& elfun,
  mfem::DenseMatrix& elmat)
{
  int dof = el.GetDof(), dim = el.GetDim();

  DSh.SetSize(dof, dim);
  DS.SetSize(dof, dim);
  Jrt.SetSize(dim);
  Jpt.SetSize(dim);
  PMatI.UseExternalData(elfun.GetData(), dof, dim);
  elmat.SetSize(dof * dim);

  const mfem::IntegrationRule* ir = IntRule;
  if (!ir) {
    ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3));  // <---
  }

  elmat = 0.0;
  model->SetTransformation(Ttr);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir->IntPoint(i);
    Ttr.SetIntPoint(&ip);
    CalcInverse(Ttr.Jacobian(), Jrt);

    el.CalcDShape(ip, DSh);
    Mult(DSh, Jrt, DS);
    MultAtB(PMatI, DS, Jpt);

    for (int d = 0; d < dim; d++) {
      Jpt(d, d) += 1.0;
    }

    model->AssembleH(Jpt, DS, ip.weight * Ttr.Weight(), elmat);
  }
}


NonlinearSolidQuasiStaticOperator::NonlinearSolidQuasiStaticOperator(std::shared_ptr<mfem::NonlinearForm> H_form)
  : mfem::Operator(H_form->FESpace()->GetTrueVSize())
{
  m_H_form = H_form;
}

// compute: y = H(x,p)
void NonlinearSolidQuasiStaticOperator::Mult(const mfem::Vector& k, mfem::Vector& y) const
{
  // Apply the nonlinear form
  m_H_form->Mult(k, y);
}

// Compute the Jacobian from the nonlinear form
mfem::Operator& NonlinearSolidQuasiStaticOperator::GetGradient(const mfem::Vector& x) const
{
  return m_H_form->GetGradient(x);
}

// destructor
NonlinearSolidQuasiStaticOperator::~NonlinearSolidQuasiStaticOperator() {}


void HyperelasticTractionIntegrator::AssembleFaceVector(const mfem::FiniteElement& el1,
  const mfem::FiniteElement& el2,
  mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
  mfem::Vector& elvec)
{
  int dim = el1.GetDim();
  int dof = el1.GetDof();

  shape.SetSize(dof);
  elvec.SetSize(dim * dof);

  DSh_u.SetSize(dof, dim);
  DS_u.SetSize(dof, dim);
  J0i.SetSize(dim);
  F.SetSize(dim);
  Finv.SetSize(dim);

  PMatI_u.UseExternalData(elfun.GetData(), dof, dim);

  int                          intorder = 2 * el1.GetOrder() + 3;
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(Tr.FaceGeom, intorder);

  elvec = 0.0;

  mfem::Vector trac(dim);
  mfem::Vector ftrac(dim);
  mfem::Vector nor(dim);
  mfem::Vector fnor(dim);
  mfem::Vector u(dim);
  mfem::Vector fu(dim);

  for (int i = 0; i < ir.GetNPoints(); i++) {
    const mfem::IntegrationPoint& ip = ir.IntPoint(i);
    mfem::IntegrationPoint        eip;
    Tr.Loc1.Transform(ip, eip);

    Tr.Face->SetIntPoint(&ip);

    CalcOrtho(Tr.Face->Jacobian(), nor);

    // Normalize vector
    double norm = nor.Norml2();
    nor /= norm;

    // Compute traction
    function.Eval(trac, *Tr.Face, ip);

    Tr.Elem1->SetIntPoint(&eip);
    CalcInverse(Tr.Elem1->Jacobian(), J0i);

    el1.CalcDShape(eip, DSh_u);
    Mult(DSh_u, J0i, DS_u);
    MultAtB(PMatI_u, DS_u, F);

    for (int d = 0; d < dim; d++) {
      F(d, d) += 1.0;
    }

    CalcInverse(F, Finv);

    Finv.MultTranspose(nor, fnor);

    el1.CalcShape(eip, shape);
    for (int j = 0; j < dof; j++) {
      for (int k = 0; k < dim; k++) {
        elvec(dof * k + j) -= trac(k) * shape(j) * ip.weight * Tr.Face->Weight() * F.Det() * fnor.Norml2();
      }
    }
  }
}

void HyperelasticTractionIntegrator::AssembleFaceGrad(const mfem::FiniteElement& el1,
  const mfem::FiniteElement& el2,
  mfem::FaceElementTransformations& Tr, const mfem::Vector& elfun,
  mfem::DenseMatrix& elmat)
{
  double       diff_step = 1.0e-8;
  mfem::Vector temp_out_1;
  mfem::Vector temp_out_2;
  mfem::Vector temp(elfun.GetData(), elfun.Size());

  elmat.SetSize(elfun.Size(), elfun.Size());

  for (int j = 0; j < temp.Size(); j++) {
    temp[j] += diff_step;
    AssembleFaceVector(el1, el2, Tr, temp, temp_out_1);
    temp[j] -= 2.0 * diff_step;
    AssembleFaceVector(el1, el2, Tr, temp, temp_out_2);

    for (int k = 0; k < temp.Size(); k++) {
      elmat(k, j) = (temp_out_1[k] - temp_out_2[k]) / (2.0 * diff_step);
    }
    temp[j] = elfun[j];
  }
}
