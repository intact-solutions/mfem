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

// Initial condition
void omega_function(const Vector& x, Vector &y) {
  y.SetSize(2);
  y[0] = x[0];
  y[1] = 1; //derivitive of x
};


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
  };  
};

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.   
   int order = 1;
   bool static_cond = false;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   // 2. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 3. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh *mesh = new Mesh(10,1.0);
   int dim = mesh->Dimension();  

   // 5. Define a finite element space on the mesh. Here we use continuous
   //    Lagrange finite elements of the specified order. If order < 1, we
   //    instead use an isoparametric/isogeometric space.
   FiniteElementCollection *fec;
   fec = new H1_FECollection(order = 1, dim);

   FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);
   cout << "Number of finite element unknowns: "
        << fespace->GetTrueVSize() << endl;
   int size = fespace->GetVSize();

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   //    In this example, the boundary conditions are defined by marking only
   //    boundary attribute 1 from the mesh as essential and converting it to a
   //    list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   //fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   cout << "\nessential dofs: ";
   ess_tdof_list.Print();

  
   Vector rhs(size);
   rhs = 0.0;   
   rhs[size-1] = 5.0;
   cout << "\nRHS: ";
   rhs.Print();
   
   ConstantCoefficient one(1.0);
   
   // 8. Define the solution vector x as a finite element grid function
   //    corresponding to fespace. Initialize x with initial guess of zero,
   //    which satisfies the boundary conditions.
   GridFunction x(fespace);
   x = 0.0;

   // 9. Set up the bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   VectorFunctionCoefficient omega(2, omega_function);

   BilinearForm *a = new BilinearForm(fespace);
   if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   a->AddDomainIntegrator(new DiffusionIntegrator_Omega(one, 2, omega));

   // 10. Assemble the bilinear form and the corresponding linear system,
   //     applying any necessary transformations such as: eliminating boundary
   //     conditions, applying conforming constraints for non-conforming AMR,
   //     static condensation, etc.
   if (static_cond) { a->EnableStaticCondensation(); }
   a->Assemble();

   OperatorPtr A;
   Vector B, X;
   a->FormLinearSystem(ess_tdof_list, x, rhs, A, X, B);

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
   a->RecoverFEMSolution(X, rhs, x);
   //cout << "\nSolution: "; x.Print();
    
   GridFunction grad_x(fespace);
   grad_x = 0.0;
   x.GetDerivative(1, 0, grad_x);
   //cout << "\nGrad X"; grad_x.Print();

   //apply solution structure
   GridFunction omega_gf(fespace), omegagrad_gf(fespace);
   omega_gf = 0.0; omegagrad_gf = 0.0;

   GridFunction x_omega(fespace), gradx_omega(fespace);
   x_omega = 0.0; gradx_omega = 0.0;

   auto x_data = x.GetData();
   auto gradx_data = grad_x.GetData();
   auto x_omega_data = x_omega.GetData();
   auto gradx_omega_data = gradx_omega.GetData();

   for (int i = 0; i < x.Size(); i++) {
     //query omega
     Vector point_vec(1), d_dx(2);
     point_vec(0) = mesh->GetVertex(i)[0];
     omega_function(point_vec, d_dx);
     
     //change the solution and it's grad
     x_omega_data[i] = x_data[i] * d_dx[0];
     gradx_omega_data[i] = gradx_data[i] * d_dx[0] + x_data[i] * d_dx[1];

     //store the omega and omega_grad
     omega_gf(i) = d_dx[0];
     omegagrad_gf(i) = d_dx[1];
   }


   ofstream vtk_ofs("D:\\OneDrive\\Documents\\VisualStudio2017\\Projects\\mfem\\examples\\ex1_sol.vtk");
   mesh->PrintVTK(vtk_ofs, 1, 0);
   x.SaveVTK(vtk_ofs, "sol_wo_omega", 1);
   grad_x.SaveVTK(vtk_ofs, "sol_grad_wo_omega", 1);
   x_omega.SaveVTK(vtk_ofs, "sol", 1);
   gradx_omega.SaveVTK(vtk_ofs, "sol_grad", 1);
   omega_gf.SaveVTK(vtk_ofs, "omega", 1);
   omegagrad_gf.SaveVTK(vtk_ofs, "omega_grad", 1);
      
   // 15. Free the used memory.
   delete a;   
   delete fespace;
   if (order > 0) { delete fec; }
   delete mesh;

   return 0;
}