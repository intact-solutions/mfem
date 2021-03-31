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

double omega_cutoff = 1;
int omega_order = 1;
double applied_load = 5.0;
double geometry_length = 1.0;
bool smoothed_flag = false;

void omega_function(const Vector& x, Vector &y) {
  y.SetSize(2);
  
  if (smoothed_flag) {
    if (x[0] <= omega_cutoff) {
      y[0] = -pow(x[0], 3) + 0.5 * pow(x[0], 4) + x[0];
      y[1] = -3 * pow(x[0], 2) + 2 * pow(x[0], 3) + 1;
    }
    else {
      y[0] = 0.5;
      y[1] = 0;
    }
  }
  else {
    if (x[0] <= omega_cutoff) {
      double exp = 1 - x[0] / omega_cutoff;
      y[0] = omega_cutoff / omega_order * (1 - pow(exp, omega_order));
      y[1] = pow(exp, omega_order - 1);
    }
    else {
      y[0] = omega_cutoff / omega_order;
      y[1] = 0;
    }
  }
 };

// Initial condition
double omega_only_function(const Vector& x) {
  if (smoothed_flag) {
    if (x[0] <= omega_cutoff) {
      return -pow(x[0], 3) + 0.5 * pow(x[0], 4) + x[0];     
    }
    else {
      return 0.5;
    }
  }
  else {
    if (x[0] <= omega_cutoff) {
      double exp = 1 - x[0] / omega_cutoff;
      return omega_cutoff / omega_order * (1 - pow(exp, omega_order));
    }
    else {
      return omega_cutoff / omega_order;
    }
  }
};

double omegagrad_only_function(const Vector& x) {
  if (smoothed_flag) {
    if (x[0] <= omega_cutoff) {
      return -3 * pow(x[0], 2) + 2 * pow(x[0], 3) + 1;
    }
    else {
      return 0;
    }
  }
  else {
    if (x[0] <= omega_cutoff) {
      double exp = 1 - x[0] / omega_cutoff;
      return pow(exp, omega_order - 1);
    }
    else {
      return 0;
    }
  }
};

double exact_solution_wo(const Vector& x) {
  double omega = 0.0;
  if (x[0] <= omega_cutoff) {
    double exp = 1 - x[0] / omega_cutoff;
    omega = omega_cutoff / omega_order * (1 - pow(exp, omega_order));
  }
  else {
    omega = omega_cutoff / omega_order;
  }

  double exact_sol = x[0] * applied_load / geometry_length;
  double limit = 1e-10;
  if (x[0] < limit) { // limit behavior
    exact_sol = limit * applied_load / geometry_length;
    double exp = 1 - limit / omega_cutoff;
    omega = omega_cutoff / omega_order * (1 - pow(exp, omega_order));
    return exact_sol / omega;
  }
  else return exact_sol / omega;
}

/** Class for integrating the bilinear form a(u,v) := (Q grad u, grad v) where Q
    can be a scalar or a matrix coefficient. */
class DiffusionIntegrator_Omega : public BilinearFormIntegrator
{
protected:
  Coefficient* Q;
  MatrixCoefficient* MQ;
  int intorder = 1;
  VectorFunctionCoefficient* omega;

private:
  Vector vec, pointflux, shape;
#ifndef MFEM_THREAD_SAFE
  DenseMatrix dshape, dshapedxt, invdfdx, mq;
  DenseMatrix te_dshape, te_dshapedxt;
#endif

  // PA extension
  const FiniteElementSpace* fespace;
  const DofToQuad* maps;         ///< Not owned
  const GeometricFactors* geom;  ///< Not owned
  int dim, ne, dofs1D, quad1D;
  Vector pa_data;

#ifdef MFEM_USE_CEED
  // CEED extension
  CeedData* ceedDataPtr;
#endif

public:
  /// Construct a diffusion integrator with coefficient Q = 1
  DiffusionIntegrator_Omega()
  {
    Q = NULL;
    MQ = NULL;
    maps = NULL;
    geom = NULL;
    omega = NULL;
#ifdef MFEM_USE_CEED
    ceedDataPtr = NULL;
#endif
  }

  /// Construct a diffusion integrator with a scalar coefficient q
  DiffusionIntegrator_Omega(Coefficient& q, int order, VectorFunctionCoefficient& omega_in)
    : Q(&q), intorder(order), omega(&omega_in)
  {
    MQ = NULL;
    maps = NULL;
    geom = NULL;
#ifdef MFEM_USE_CEED
    ceedDataPtr = NULL;
#endif
  }

 /* /// Construct a diffusion integrator with a matrix coefficient q
  DiffusionIntegrator_Omega(MatrixCoefficient& q, int order, FunctionCoefficient omega_in)
    : MQ(&q), intorder(order), omega(omega_in)
  {
    Q = NULL;
    maps = NULL;
    geom = NULL;
#ifdef MFEM_USE_CEED
    ceedDataPtr = NULL;
#endif
  }*/

  virtual ~DiffusionIntegrator_Omega()
  {
#ifdef MFEM_USE_CEED
    delete ceedDataPtr;
#endif
  }

  /** Given a particular Finite Element
      computes the element stiffness matrix elmat. */
  virtual void AssembleElementMatrix(const FiniteElement& el,
    ElementTransformation& Trans,
    DenseMatrix& elmat) {
    int nd = el.GetDof();
    int dim = el.GetDim();
    int spaceDim = Trans.GetSpaceDim();
    bool square = (dim == spaceDim);
    double w, jacobian;

    dshape.SetSize(nd, dim);
    dshapedxt.SetSize(nd, spaceDim);
    invdfdx.SetSize(dim, spaceDim);
    shape.SetSize(nd);
    elmat.SetSize(nd);

    const IntegrationRule ir = IntRules.Get(el.GetGeomType(), intorder);

    elmat = 0.0;
    for (int i = 0; i < ir.GetNPoints(); i++)
    {
      const IntegrationPoint& ip = ir.IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);
      w = Trans.Weight();
      w = ip.weight / (square ? w : w * w * w);
      jacobian = Trans.Jacobian().Elem(0,0);

      Vector d_dx;
      omega->Eval(d_dx, Trans, ip);

      //scale the omega derivitve to reference space
      d_dx[1] = d_dx[1] * jacobian;

      //modify dshape to include omega
      auto dshape_data = dshape.Data();
      for (int i = 0; i < shape.Size(); i++) {
        // Liebniz rule
        dshape_data[i] = dshape_data[i] * d_dx[0] +
          shape[i] * d_dx[1];
      }
      
      // AdjugateJacobian = / adj(J),         if J is square
      //                    \ adj(J^t.J).J^t, otherwise
      Mult(dshape, Trans.AdjugateJacobian(), dshapedxt);
      if (!MQ)
      {
        if (Q)
        {
          w *= Q->Eval(Trans, ip);
        }
        AddMult_a_AAt(w, dshapedxt, elmat);
      }
      else
      {
        MQ->Eval(invdfdx, Trans, ip);
        invdfdx *= w;
        Mult(dshapedxt, invdfdx, dshape);
        AddMultABt(dshape, dshapedxt, elmat);
      }
    }
    //cout << "\nelamt: ";  elmat.Print();
  };  
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.   
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   int element_order = 1;
   omega_order = 3;   
   int integration_order = ( omega_order + element_order + 1); //2n-1 --> accuracy of guassian quadrature
   cout << "\nintegration order: " << integration_order;

   smoothed_flag = true;
   int num_elements = 25;
   geometry_length = 10.0;
   omega_cutoff = 1.0;   
   int sampling_mesh_ref = 10;

   std::vector<double> vec = {10,20,50,70,100};

   // 2. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh* mesh = new Mesh(num_elements, geometry_length);
   int dim = mesh->Dimension();

   ofstream vtk_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\ex1_sol.vtk");
   mesh->PrintVTK(vtk_ofs, sampling_mesh_ref, 0);

   for (auto elem_i : vec) {
     num_elements = elem_i;
     std::string error_name("num_elements");

     // 5. Define a finite element space on the mesh. Here we use continuous
     //    Lagrange finite elements of the specified order. If order < 1, we
     //    instead use an isoparametric/isogeometric space.
     FiniteElementCollection* fec;
     fec = new H1_FECollection(element_order, dim);

     FiniteElementSpace* fespace = new FiniteElementSpace(mesh, fec);
     cout << "\nNumber of finite element unknowns: "
       << fespace->GetTrueVSize() << endl;
     int size = fespace->GetVSize();

     // 6. Determine the list of true (i.e. conforming) essential boundary dofs.   
     Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
     ess_bdr = 0;
     ess_bdr[0] = 1;
     cout << "\nessential dofs: ";
     ess_tdof_list.Print();

     //assign load after modifying it by solution structure
     Vector point_vec(1), d_dx(2);
     point_vec(0) = geometry_length;
     omega_function(point_vec, d_dx);

     Vector rhs(size);
     rhs = 0.0;
     rhs[mesh->GetNV() - 1] = applied_load * d_dx[0]; //use the index from mesh as fespace indexing is not linear in x
     cout << "\nRHS: ";
     rhs.Print();

     ConstantCoefficient one(1.0);

     // 8. Define the solution vector x as a finite element grid function
     GridFunction x(fespace);
     x = 0.0;

     // 9. Set up the bilinear form a(.,.) 
     VectorFunctionCoefficient omega(2, omega_function);

     BilinearForm* a = new BilinearForm(fespace);
     if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
     a->AddDomainIntegrator(new DiffusionIntegrator_Omega(one, integration_order, omega));

     // 10. Assemble the bilinear form 
     if (static_cond) { a->EnableStaticCondensation(); }
     a->Assemble();

     OperatorPtr A;
     Vector B, X;
     a->FormLinearSystem(ess_tdof_list, x, rhs, A, X, B);

     cout << "Size of linear system: " << A->Height() << endl;

     // 11. Solve the linear system A X = B.
     if (!pa)
     {
       // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
       GSSmoother M((SparseMatrix&)(*A));
       PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
     }
     else // Jacobi preconditioning in partial assembly mode
     {
       OperatorJacobiSmoother M(*a, ess_tdof_list);
       PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
     }

     // 12. Recover the solution as a finite element grid function.
     a->RecoverFEMSolution(X, rhs, x);

     GridFunction grad_x(fespace);
     grad_x = 0.0;
     x.GetDerivative(1, 0, grad_x);

     //apply solution structure
     GridFunction omega_gf(fespace), omegagrad_gf(fespace);
     omega_gf = 0.0; omegagrad_gf = 0.0;

     //obtain omega and omega grad using projection
     FunctionCoefficient omega_only(omega_only_function);
     omega_gf.ProjectCoefficient(omega_only);

     FunctionCoefficient omegagrad_only(omegagrad_only_function);
     omegagrad_gf.ProjectCoefficient(omegagrad_only);

     //solution with solution structure
     GridFunction x_omega(fespace), gradx_omega(fespace);
     x_omega = 0.0; gradx_omega = 0.0;

     //exact solution without solution structure
     GridFunction x_exact_wo(fespace), gradx_exact_wo(fespace);
     x_exact_wo = 0.0; gradx_exact_wo = 0.0;
     FunctionCoefficient f_ex( exact_solution_wo);
     x_exact_wo.ProjectCoefficient(f_ex);

     //exact solution with solution structure
     GridFunction x_exact(fespace), gradx_exact(fespace);
     x_exact = 0.0; gradx_exact = 0.0;

     //error
     GridFunction grad_error(fespace); GridFunction zero(fespace);
     grad_error = 0.0; zero = 0.0;

     auto x_data = x.GetData();
     auto gradx_data = grad_x.GetData();
     auto x_omega_data = x_omega.GetData();
     auto gradx_omega_data = gradx_omega.GetData();

     for (int i = 0; i < x.Size(); i++) {       
       //change the solution and it's grad
       x_omega_data[i] = x_data[i] * omega_gf(i);
       gradx_omega_data[i] = gradx_data[i] * omega_gf(i) + x_data[i] * omegagrad_gf(i);     

       //exact solution
       x_exact[i] = point_vec(0) * applied_load;
       gradx_exact[i] = applied_load;

       //error
       grad_error[i] = (gradx_exact[i] - gradx_omega_data[i]) / gradx_exact[i] * 100.0;
     }
    
     //x.SaveVTK(vtk_ofs, "sol_wo", sampling_mesh_ref);
     //grad_x.SaveVTK(vtk_ofs, "sol_grad_wo", sampling_mesh_ref);
     //x_omega.SaveVTK(vtk_ofs, "sol", sampling_mesh_ref);
     //gradx_omega.SaveVTK(vtk_ofs, "sol_grad", sampling_mesh_ref);
     //omega_gf.SaveVTK(vtk_ofs, "omega", sampling_mesh_ref);
     //omegagrad_gf.SaveVTK(vtk_ofs, "omega_grad", sampling_mesh_ref);
     //x_exact_wo.SaveVTK(vtk_ofs, "sol_wo_exact", sampling_mesh_ref);
     ////gradx_exact_wo.SaveVTK(vtk_ofs, "sol_grad_wo_exact", sampling_mesh_ref);
     //x_exact.SaveVTK(vtk_ofs, "sol_exact", sampling_mesh_ref);
     //gradx_exact.SaveVTK(vtk_ofs, "sol_grad_exact", sampling_mesh_ref);

     std::stringstream ss;
     ss << std::fixed << std::setprecision(2) << elem_i;
     std::string mystring = ss.str();

    
     error_name = error_name + "_" + mystring;

     grad_error.SaveVTK(vtk_ofs, error_name, sampling_mesh_ref);
     //zero.SaveVTK(vtk_ofs, "zero", sampling_mesh_ref);

     std::ofstream fout("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\error.txt");
     grad_error.Save(fout);


     // 15. Free the used memory.
     delete a;
     delete fespace;
     if (element_order > 0) { delete fec; }     
   }
   delete mesh;
   return 0;
}