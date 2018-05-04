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
    if (load_face) {
      bdrface->SetAttribute(force_bdratt);
      //auto vertices = bdrface->GetVertices();
      //cout << "loaded vertices: " << vertices[0] << ", " << vertices[2] << ", " << vertices[2] << "\n";
    }
  }

  {
    ofstream mesh_ofs("block.vtk");
    mesh_ofs.precision(8);
    mesh->PrintVTK(mesh_ofs);
  }

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 1000
  //    elements.
  {
    int ref_levels = 2;
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

 //sample ply mesh
  GridFunction x_ply(fespace_ply);
  x_ply = 0.0;
  double * ply_data = x_ply.GetData();
  int ply_data_size = x_ply.Size();
  cout << "Size " << ply_data_size << endl;
  int index = 0;
  for (int i = 0; i < plymesh->GetNV(); i++) {
    Vector point(plymesh->GetVertex(i),3);
    DenseMatrix points(dim, 1);
    points.SetCol(0, point);
    Array<int> elem_ids;
    Array<IntegrationPoint> ips;
    mesh->FindPoints(points, elem_ids, ips);
    cout << "Point: " << point[0] << " " << point[1] << " " << point[2] << " ";
    cout << "Element Id: " << elem_ids[0] << " size " << elem_ids.Size() << " relative point " << ips[0].x << ", " << ips[0].y << ", " << ips[0].z << " ";
    //cout << x.GetValue(elem_ids[0], ips[0], 0) << ", " << x.GetValue(elem_ids[0], ips[0]) <<", " << x.GetValue(elem_ids[0], ips[0],2)  <<endl;

    Array<int> dof(1), vdof(3);
    dof = -1;
    vdof = -1;
    fespace_ply->GetVertexDofs(i, dof);
    cout << "dof: "<< dof[0] << ", vdof: " << fespace_ply->DofToVDof(dof[0], 0) << ", " << fespace_ply->DofToVDof(dof[0], 1) << ", " << fespace_ply->DofToVDof(dof[0], 2) << " ";

    cout << "Displacement value [" << x.GetValue(elem_ids[0], ips[0], 0) << ", " << x.GetValue(elem_ids[0], ips[0]) <<", " << x.GetValue(elem_ids[0], ips[0],2)  <<"]\n";
    ply_data[fespace_ply->DofToVDof(dof[0], 0)] = x.GetValue(elem_ids[0], ips[0], 1);
    ply_data[fespace_ply->DofToVDof(dof[0], 1)] = x.GetValue(elem_ids[0], ips[0], 2);
    ply_data[fespace_ply->DofToVDof(dof[0], 2)] = x.GetValue(elem_ids[0], ips[0], 3);
  }


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