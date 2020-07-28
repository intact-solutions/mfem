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


#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

/*example 1
 * Solve Nonlinear Problem: -Laplace u + u^2 - f = 0
 * Exact solution u_exact = sin(c*pi*x)
 *
 * Newton Linearization: Given u, solve du
 *-Laplace du + 2u*du = - F(u), where F(u) = -Laplace u + u^2 - f
 * */

/*example 2
* Solve Nonlinear Problem: -grad(u grad(u)) - f = 0
* Exact solution u_exact = sin(c*pi*x)
* */

/*example 3
* Solve Nonlinear Problem: -grad(u grad(u)) - f = 0
* Exact solution u_exact = x(1-x)
*/

/*example 4 (Thermal problem)
* Solve Nonlinear Problem: -grad(u grad(u)) - f = 0
* One end at 0K, Another end has flux of 1 unit applied
* Exact solution u_exact = unkown
*/

//non-linear 3D thermal problem test
double alpha = 1.0e-1;
double kappa = 0.3;
int c;
int example;
// u = sin( c *  pi * x)
double u_exact_(const Vector& x)
{
  MFEM_ASSERT(x.Size() == 1, "Must be 1D mesh");

  if (example == 3)
    return x[0] * (1 - x[0]);

  else if (example == 4)
    return sqrt(2 * x[0]);

  else 
    return sin(c * M_PI * x[0]);
}


double f_exact_(const Vector& x)
{
  if (example == 1)// -Laplace u + u^2 = f, deduced from analytic solution u_exact
    return c * c * M_PI * M_PI * sin(c * M_PI * x[0]) + sin(c * M_PI * x[0]) * sin(c * M_PI * x[0]);
  else if (example == 2)// -div(u du) = f, deduced from analytic solution u_exact
    return c * c * M_PI * M_PI
    * (sin(c * M_PI * x[0]) * sin(c * M_PI * x[0])
      - cos(c * M_PI * x[0]) * cos(c * M_PI * x[0]));
  else if (example == 3)
    return -1 * (6 * x[0] * x[0] - 6 * x[0] + 1);
  else
    return 0;
}


class NLFIntegrator_Coeff : public NonlinearFormIntegrator
{
private:
  Vector shape;
  DenseMatrix dshape, dshapedxt, invdfdx;
  Vector vec, pointflux;
  Coefficient* f;
public:
  NLFIntegrator_Coeff(Coefficient& Q_) :f(&Q_) {}
  virtual void AssembleElementVector(const FiniteElement& el,
    ElementTransformation& Tr,
    const Vector& elfun,
    Vector& elvect)
  {
    int dof = el.GetDof();
    int dim = el.GetDim();
    shape.SetSize(dof);
    dshape.SetSize(dof, dim);
    dshapedxt.SetSize(dof, dim);
    invdfdx.SetSize(dim);
    vec.SetSize(dim);
    pointflux.SetSize(dim);
    double w;

    elvect.SetSize(dof);
    elvect = 0.0;

    //std::cout << "\nu coeff from last iteration: "; elfun.Print();

    const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), invdfdx); // invdfdx = adj(J)
      w = ip.weight / Tr.Weight();

      dshape.MultTranspose(elfun, vec);
      invdfdx.MultTranspose(vec, pointflux);

      double a0 = elfun * shape;
      w *= a0;      

      pointflux *= w;      
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt); //
      dshapedxt.AddMult(pointflux, elvect);
     /* std::cout << "\ndshape: "; dshape.Print();
      std::cout << "\ndshapedxt: "; dshapedxt.Print();
      std::cout << "\nelem vector after adding diffusion: "; elvect.Print();*/
      //Given u, compute (-f, v), v is shape function or \integration (-f)*shape 
      double fun_val =  - (*f).Eval(Tr, ip);      
      w = ip.weight * Tr.Weight();
      add(elvect, w * fun_val, shape, elvect);
      //std::cout << "\nelem vector after adding load rhs: "; elvect.Print();
    }
  }

  virtual void AssembleElementGrad(const FiniteElement& el,
    ElementTransformation& Tr,
    const Vector& elfun,
    DenseMatrix& elmat) {
    int dof = el.GetDof();
    int dim = el.GetDim();
    dshapedxt.SetSize(dof, dim);
    dshape.SetSize(dof, dim);
    shape.SetSize(dof);
    elmat.SetSize(dof);
    elmat = 0.0;

    const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);
      Tr.SetIntPoint(&ip);

      // Compute (a0 * grad(du), grad(v)).  Ref: DiffusionIntegrator::AssembleElementMatrix()
      double w = ip.weight / Tr.Weight();
      double a0 = elfun * shape;
      w *= a0;
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt); //
      AddMult_a_AAt(w, dshapedxt, elmat);

      // Compute (da0 u grad(u_o), grad(v)).  Ref: DiffusionIntegrator::AssembleElementMatrix()
      double da0 = 1; //d/du(u) = 1
      w = ip.weight / (Tr.Weight()); //reset weight
      w *= da0;
      dshape.MultTranspose(elfun, vec);
      invdfdx.MultTranspose(vec, pointflux);
      pointflux *= w;     
            
      DenseMatrix first_term(dim, dof);      
      MultVWt(pointflux, shape, first_term);
      AddMult(dshapedxt, first_term, elmat);
      //note for above opeation: doing the transverse seems correct from the forumuylation, but didn't work. So, MultVWt(shape, pointflux, first_term); and AddMult(first_term, dshapedxt, elmat) didn't work for example 2, worked for example 3, but took a lot more iterations
    }
  }
};

class NLFIntegrator_Thermal : public NonlinearFormIntegrator
{
private:
  Vector shape;
  DenseMatrix dshape, dshapedxt, invdfdx;
  Vector vec, pointflux;
  Coefficient* f;
public:
  NLFIntegrator_Thermal(Coefficient& Q_) :f(&Q_) {}
  virtual void AssembleElementVector(const FiniteElement& el,
    ElementTransformation& Tr,
    const Vector& elfun,
    Vector& elvect)
  {
    int dof = el.GetDof();
    int dim = el.GetDim();
    shape.SetSize(dof);
    dshape.SetSize(dof, dim);
    dshapedxt.SetSize(dof, dim);
    invdfdx.SetSize(dim);
    vec.SetSize(dim);
    pointflux.SetSize(dim);
    double w;

    elvect.SetSize(dof);
    elvect = 0.0;

    //std::cout << "\nu coeff from last iteration: "; elfun.Print();

    const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), invdfdx); // invdfdx = adj(J)
      w = ip.weight / Tr.Weight();

      dshape.MultTranspose(elfun, vec);
      invdfdx.MultTranspose(vec, pointflux);

      double a0_t = elfun * shape;
      double a0 = kappa + alpha * a0_t; //kappa + alpha * u
      w *= a0;

      pointflux *= w;
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt); //
      dshapedxt.AddMult(pointflux, elvect);
      /* std::cout << "\ndshape: "; dshape.Print();
       std::cout << "\ndshapedxt: "; dshapedxt.Print();
       std::cout << "\nelem vector after adding diffusion: "; elvect.Print();*/
       //Given u, compute (-f, v), v is shape function or \integration (-f)*shape 
      double fun_val = -(*f).Eval(Tr, ip);
      w = ip.weight * Tr.Weight();
      add(elvect, w * fun_val, shape, elvect);
      //std::cout << "\nelem vector after adding load rhs: "; elvect.Print();
    }
  }

  virtual void AssembleElementGrad(const FiniteElement& el,
    ElementTransformation& Tr,
    const Vector& elfun,
    DenseMatrix& elmat) {
    int dof = el.GetDof();
    int dim = el.GetDim();
    dshapedxt.SetSize(dof, dim);
    dshape.SetSize(dof, dim);
    shape.SetSize(dof);
    elmat.SetSize(dof);
    elmat = 0.0;

    const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);
      Tr.SetIntPoint(&ip);

      // Compute (a0 * grad(du), grad(v)).  Ref: DiffusionIntegrator::AssembleElementMatrix()
      double w = ip.weight / Tr.Weight();
      double a0_t = elfun * shape;
      double a0 = kappa + alpha * a0_t; //kappa + alpha * u
      w *= a0;
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt); //
      AddMult_a_AAt(w, dshapedxt, elmat);

      // Compute (da0 u grad(u_o), grad(v)).  Ref: DiffusionIntegrator::AssembleElementMatrix()
      double da0 = alpha; //d/du(u) = 1
      w = ip.weight / (Tr.Weight()); //reset weight
      w *= da0;
      dshape.MultTranspose(elfun, vec);
      invdfdx.MultTranspose(vec, pointflux);
      pointflux *= w;

      DenseMatrix first_term(dim, dof);
      MultVWt(pointflux, shape, first_term);
      AddMult(dshapedxt, first_term, elmat);
      //note for above opeation: doing the transverse seems correct from the forumuylation, but didn't work. So, MultVWt(shape, pointflux, first_term); and AddMult(first_term, dshapedxt, elmat) didn't work for example 2, worked for example 3, but took a lot more iterations
    }
  }
};

class NLFIntegrator : public NonlinearFormIntegrator
{
private:
  Vector shape;
  DenseMatrix dshape, dshapedxt, invdfdx;
  Vector vec, pointflux;
  Coefficient* f;
public:
  NLFIntegrator(Coefficient &Q_) :f(&Q_) {}
  virtual void AssembleElementVector(const FiniteElement& el,
    ElementTransformation& Tr,
    const Vector& elfun,
    Vector& elvect)
  {
    int dof = el.GetDof();
    int dim = el.GetDim();
    shape.SetSize(dof);
    dshape.SetSize(dof, dim);
    invdfdx.SetSize(dim);
    vec.SetSize(dim);
    pointflux.SetSize(dim);
    double w;

    elvect.SetSize(dof);
    elvect = 0.0;
    
    const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Tr.SetIntPoint(&ip);
      CalcAdjugate(Tr.Jacobian(), invdfdx); // invdfdx = adj(J)
      w = ip.weight / Tr.Weight();

      dshape.MultTranspose(elfun, vec);
      invdfdx.MultTranspose(vec, pointflux);
     /* if (Q)
      {
        w *= Q->Eval(Tr, ip);
      }*/
           
      pointflux *= w;      
      dshapedxt.AddMult(pointflux, elvect);
      //std::cout << "elem vector after adding diffusion: "; elvect.Print();
      //Given u, compute (u^2-f, v), v is shape function
      //or \integration (u^2-f)*shape 
      //double fun_val = -1; //4 * M_PI * M_PI * (elfun * shape); //the last product computes the current solution
      double fun_val = (elfun * shape) * (elfun * shape) - (*f).Eval(Tr, ip);
      //w = ip.weight / Tr.Weight() * fun_val;
      w = ip.weight * Tr.Weight();
      add(elvect, w * fun_val, shape, elvect);      
    }
  } 

  virtual void AssembleElementGrad(const FiniteElement& el,
    ElementTransformation& Tr,
    const Vector& elfun,
    DenseMatrix& elmat) {
    int dof = el.GetDof();
    int dim = el.GetDim();
    dshapedxt.SetSize(dof, dim);
    dshape.SetSize(dof, dim);
    shape.SetSize(dof);
    elmat.SetSize(dof);
    elmat = 0.0;

    const IntegrationRule* ir = &IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + Tr.OrderW());

    for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint& ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);
      el.CalcDShape(ip, dshape);
      Tr.SetIntPoint(&ip);

      // Compute (grad(du), grad(v)).  Ref: DiffusionIntegrator::AssembleElementMatrix()
      double w = ip.weight / Tr.Weight();
      Mult(dshape, Tr.AdjugateJacobian(), dshapedxt); //
      AddMult_a_AAt(w, dshapedxt, elmat);

      // Compute 2*u*(du,v), v is shape function
      double fun_val = 2 * (elfun * shape) * ip.weight * Tr.Weight(); // 2*u
      AddMult_a_VVt(fun_val, shape, elmat); // 2*u*(du, v)
    }
  }
};

class NLOperator : public Operator
{
private:
  NonlinearForm* N;
  mutable SparseMatrix* Jacobian;  

public:
  NLOperator(NonlinearForm* N_, int size) : Operator(size), N(N_), Jacobian(NULL) { }

  virtual void Mult(const Vector& x, Vector& y) const
  {  
    N->Mult(x, y); //Evaluate the action of the NonlinearForm
    //y.Neg();
    /*cout << "\nIn NLOperator (after)";
    cout << "\nx print: "; x.Print();
    cout << "\ny print: "; y.Print();*/
  }

  virtual Operator& GetGradient(const Vector& x) const
  {
    Jacobian = dynamic_cast<SparseMatrix*>(&N->GetGradient(x));
    //std::cout << "\nJacobian: "; Jacobian->Print();
    return *Jacobian;
  }
};


int main1()
{
  c = 2;
  example = 4;
  Mesh mesh(40, 1.0);
  int dim = mesh.Dimension();

  H1_FECollection h1_fec(1, dim);
  FiniteElementSpace h1_space(&mesh, &h1_fec);
  int size = h1_space.GetVSize();

  Array<int> ess_tdof_list;
  if (mesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;
    h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    cout << "\nessential dofs: ";
    ess_tdof_list.Print();
  }
    
  FunctionCoefficient f_exact_coeff(f_exact_);
     
  NonlinearForm N(&h1_space);
  if(example == 1)
    N.AddDomainIntegrator(new NLFIntegrator(f_exact_coeff));
  else
    N.AddDomainIntegrator(new NLFIntegrator_Coeff(f_exact_coeff));

  if (example == 4) {
    //only one end is fixed
    Array<int> ess_tdof_list_4;
    ess_tdof_list_4.Append(ess_tdof_list[0]);
    N.SetEssentialTrueDofs(ess_tdof_list_4);
  }
  else
    N.SetEssentialTrueDofs(ess_tdof_list);
  
  NLOperator N_oper(&N, size);

  Solver* J_solver;
  Solver* J_prec = new DSmoother(1);
  MINRESSolver* J_minres = new MINRESSolver;
  J_minres->SetRelTol(1e-12);
  J_minres->SetAbsTol(1e-12);
  J_minres->SetMaxIter(200);
  J_minres->SetPreconditioner(*J_prec);
  J_solver = J_minres;

  NewtonSolver newton_solver;
  newton_solver.SetRelTol(1e-10);
  newton_solver.SetAbsTol(1e-10);
  newton_solver.SetMaxIter(200);
  newton_solver.SetSolver(*J_solver);
  newton_solver.SetOperator(N_oper);
  newton_solver.SetPrintLevel(1);
  newton_solver.iterative_mode = true;
    
  GridFunction uh(&h1_space);

  //for non-zero initial value (still must satisfy essential boundary condition)
  ConstantCoefficient const_coeff(1.0);
  uh.ProjectCoefficient(const_coeff);
  for(auto & e_i : ess_tdof_list)
    uh[e_i] = 0.0;

  Vector rhs(size);
  rhs = 0.0;
  if (example == 4)//right hand has flux specified
    rhs[ess_tdof_list[1]] = 1.0;

  newton_solver.Mult(rhs, uh); //solve the non-linear form with right hand side as rhs and uh has the initial guess (and will eventually store the result)

  FunctionCoefficient u_exact_coeff(u_exact_);
  //uh.ProjectCoefficient(u_exact_coeff);
  cout << "L2 error norm: " << uh.ComputeL2Error(u_exact_coeff) << endl;

  // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
  ofstream mesh_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\solution_nonlinear.vtk");
  mesh.PrintVTK(mesh_ofs, 1, 0);
  uh.SaveVTK(mesh_ofs, "u", 1);

  //ofstream mesh_ofs_ref("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\solution_reference.vtk");
  //mesh.PrintVTK(mesh_ofs_ref, 1);
  uh.ProjectCoefficient(u_exact_coeff);
  uh.SaveVTK(mesh_ofs, "u_ref", 1);
  return 0;
}

int main(int argc, char* argv[])
{
  // 1. Parse command-line options.
  const char* mesh_file = "../data/beam-tet.mesh";
  int ref_levels = 0;
  int order = 1;

  int precision = 8;
  cout.precision(precision);

  // 2. Read the mesh from the given mesh file. We can handle triangular,
  //    quadrilateral, tetrahedral and hexahedral meshes with the same code.
  Mesh mesh(mesh_file, 1, 1);
  //Mesh mesh(10, 1.0);
  int dim = mesh.Dimension();

  //Refine the mesh to increase the resolution.In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter.
  for (int lev = 0; lev < ref_levels; lev++)
  {
    mesh.UniformRefinement();
  }

  H1_FECollection h1_fec(order, dim);
  FiniteElementSpace h1_space(&mesh, &h1_fec);
  int fe_size = h1_space.GetTrueVSize();
  cout << "Number of temperature unknowns: " << fe_size << endl;

  Array<int> ess_tdof_list;
  if (mesh.bdr_attributes.Size()) {
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    h1_space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    cout << "\nessential dofs: ";
    ess_tdof_list.Print();
  }

  ConstantCoefficient f_exact_coeff(0.0);

  NonlinearForm N(&h1_space); 
  N.AddDomainIntegrator(new NLFIntegrator_Coeff(f_exact_coeff));

  ////only one end is fixed
  //Array<int> ess_tdof_list_4;
  //ess_tdof_list_4.Append(ess_tdof_list[0]);
  N.SetEssentialTrueDofs(ess_tdof_list);

  NLOperator N_oper(&N, fe_size);

  Solver* J_solver;
  Solver* J_prec = new DSmoother(1);
  MINRESSolver* J_minres = new MINRESSolver;
  J_minres->SetRelTol(1e-12);
  J_minres->SetAbsTol(1e-12);
  J_minres->SetMaxIter(200);
  J_minres->SetPreconditioner(*J_prec);
  J_solver = J_minres;

  NewtonSolver newton_solver;
  newton_solver.SetRelTol(1e-10);
  newton_solver.SetAbsTol(1e-10);
  newton_solver.SetMaxIter(200);
  newton_solver.SetSolver(*J_solver);
  newton_solver.SetOperator(N_oper);
  newton_solver.SetPrintLevel(1);
  newton_solver.iterative_mode = true;

  GridFunction uh(&h1_space);

  //for non-zero initial value (still must satisfy essential boundary condition)
  ConstantCoefficient const_coeff(1.0);
  uh.ProjectCoefficient(const_coeff);
  for (auto& e_i : ess_tdof_list)
    uh[e_i] = 0.0;

  //right hand side
  Vector surface_flux(mesh.bdr_attributes.Max());
  surface_flux = 0.0;
  surface_flux(1) = 1.0;
  PWConstCoefficient flux_coeff(surface_flux);

  LinearForm b(&h1_space);
  b.AddBoundaryIntegrator(new BoundaryLFIntegrator(flux_coeff));
  b.Assemble();
  Vector rhs(b.GetData(), fe_size);

  cout << "Rhs: "; rhs.Print();

  newton_solver.Mult(rhs, uh); //solve the non-linear form with right hand side as rhs and uh has the initial guess (and will eventually store the result)

  // 13. Save the refined mesh and the solution. This output can be viewed later
   //     using GLVis: "glvis -m refined.mesh -g sol.gf".
  ofstream mesh_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\solution_nonlinear_thermal.vtk");
  mesh.PrintVTK(mesh_ofs, 1, 0);
  uh.SaveVTK(mesh_ofs, "u", 1);

  return 0;

  
}

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

int main2(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = false;

   //OptionsParser args(argc, argv);
   //args.AddOption(&mesh_file, "-m", "--mesh",
   //               "Mesh file to use.");
   //args.AddOption(&order, "-o", "--order",
   //               "Finite element order (polynomial degree) or -1 for"
   //               " isoparametric space.");
   //args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
   //               "--no-static-condensation", "Enable static condensation.");
   //args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
   //               "--no-partial-assembly", "Enable Partial Assembly.");
   //args.AddOption(&device_config, "-d", "--device",
   //               "Device configuration string, see Device::Configure().");
   //args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
   //               "--no-visualization",
   //               "Enable or disable GLVis visualization.");
   //args.Parse();
   //if (!args.Good())
   //{
   //   args.PrintUsage(cout);
   //   return 1;
   //}
   //args.PrintOptions(cout);

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   //Mesh *mesh = new Mesh(mesh_file, 1, 1);
   Mesh *mesh = new Mesh(10, 1.0);
   int dim = mesh->Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 50,000
   //    elements.
   //{
   //   int ref_levels =
   //      (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
   //   for (int l = 0; l < ref_levels; l++)
   //   {
   //      mesh->UniformRefinement();
   //   }
   //}

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
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   std::cout << "\nFixed node";
   ess_tdof_list.Print();

   // 7. Set up the linear form b(.) which corresponds to the right-hand side of
   //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
   //    the basis functions in the finite element fespace.
   LinearForm *b = new LinearForm(fespace);
   ConstantCoefficient one(1.0);
   b->AddDomainIntegrator(new DomainLFIntegrator(one));
   b->Assemble();

   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   BilinearForm *a = new BilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator(one));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

   cout << "Size of linear system: " << A->Height() << endl;

   // 11. Solve the linear system A X = B.
   if (!pa)
   {
#ifndef MFEM_USE_SUITESPARSE
      // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
      GSSmoother M((SparseMatrix&)(*A));
      PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
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
   /*ofstream mesh_ofs("refined.mesh");
   mesh_ofs.precision(8);
   mesh->Print(mesh_ofs);
   ofstream sol_ofs("sol.gf");
   sol_ofs.precision(8);
   x.Save(sol_ofs);*/

   // 13. Save the refined mesh and the solution. This output can be viewed later
 //     using GLVis: "glvis -m refined.mesh -g sol.gf".
   ofstream mesh_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\solution_linear.vtk");
   mesh->PrintVTK(mesh_ofs, 1);
   x.SaveVTK(mesh_ofs, "u", 1);

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << *mesh << x << flush;
   }

   // 15. Free the used memory.
   delete a;
   delete b;
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}
