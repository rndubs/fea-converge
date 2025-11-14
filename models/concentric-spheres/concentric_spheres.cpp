// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Concentric Spheres Between Rigid Planes Contact Model
// Based on Figure 9 and Section 3.7 from Zimmerman & Ateshian (2018)
// "A Surface-to-Surface Finite Element Algorithm for Large Deformation
// Frictional Contact in FEBIO"
// Reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC6056201/

#include <cmath>
#include <set>
#include <string>
#include <vector>
#include <memory>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "smith/smith.hpp"

int main(int argc, char* argv[])
{
  // Initialize and automatically finalize MPI and other libraries
  smith::ApplicationManager applicationManager(argc, argv);

  // NOTE: p must be equal to 1 to work with Tribol's mortar method
  constexpr int p = 1;
  // NOTE: dim must be equal to 3
  constexpr int dim = 3;

  // Create DataStore
  std::string name = "concentric_spheres";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Geometry parameters
  const double outer_sphere_outer_radius = 2.0;
  const double outer_sphere_inner_radius = 0.7;
  const double inner_sphere_radius = 0.6;  // Slightly smaller than outer sphere inner radius

  // Material parameters (varies by friction case)
  // Case 1: mu = 0,   alpha = 1,  Ptol = 0.2
  // Case 2: mu = 0.5, alpha = 3,  Ptol = 0.2
  // Case 3: mu = 2.0, alpha = 8,  Ptol = 0.05
  const double friction_coefficient = 0.5;  // Change this to test different cases
  double penalty_parameter = 3.0;
  double ptol = 0.2;

  if (friction_coefficient == 0.0) {
    penalty_parameter = 1.0;
    ptol = 0.2;
  } else if (friction_coefficient == 2.0) {
    penalty_parameter = 8.0;
    ptol = 0.05;
  }

  // Construct inner sphere mesh (solid or thick-walled)
  // Using octant symmetry
  int num_refinements_inner{3};
  mfem::Mesh inner_sphere_mesh{SMITH_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements_inner; ++i) {
    inner_sphere_mesh.UniformRefinement();
  }
  inner_sphere_mesh.SetCurvature(p);
  inner_sphere_mesh.Transform([inner_sphere_radius](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    // Scale to desired radius
    p(0) = inner_sphere_radius * x(0);
    p(1) = inner_sphere_radius * x(1);
    p(2) = inner_sphere_radius * x(2);
  });

  // Apply octant symmetry (keep only positive octant x>=0, y>=0, z>=0)
  // TODO: This would require mesh filtering or generating octant mesh directly

  // Construct outer hollow sphere mesh
  // TODO: This requires a hollow sphere mesh with Ro=2.0, Ri=0.7
  // For now, using scaled ball mesh as placeholder
  int num_refinements_outer{3};
  mfem::Mesh outer_sphere_mesh{SMITH_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements_outer; ++i) {
    outer_sphere_mesh.UniformRefinement();
  }
  outer_sphere_mesh.SetCurvature(p);
  outer_sphere_mesh.Transform([outer_sphere_outer_radius](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = outer_sphere_outer_radius * x(0);
    p(1) = outer_sphere_outer_radius * x(1);
    p(2) = outer_sphere_outer_radius * x(2);
  });

  std::vector<mfem::Mesh*> mesh_ptrs{&inner_sphere_mesh, &outer_sphere_mesh};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
      "concentric_spheres_mesh", 0, 0);

  // Define boundary domains
  // TODO: Define symmetry planes (x=0, y=0, z=0)
  // TODO: Define contact surfaces for sphere-sphere and sphere-plane interactions

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-10,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 200,
                                                  .print_level = 1};

  // Contact options for sphere-sphere interface
  smith::ContactOptions sphere_contact_options{.method = smith::ContactMethod::SingleMortar,
                                               .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                               .type = (friction_coefficient > 0.0) ? smith::ContactType::Frictional
                                                                                    : smith::ContactType::Frictionless,
                                               .penalty = penalty_parameter,
                                               .friction_coeff = friction_coefficient,
                                               .jacobian = smith::ContactJacobian::Exact};

  // Contact options for sphere-plane interfaces (frictionless, single-pass)
  smith::ContactOptions plane_contact_options{.method = smith::ContactMethod::SingleMortar,
                                              .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                              .type = smith::ContactType::Frictionless,
                                              .penalty = 10.0,
                                              .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Zimmerman 2018 paper
  // Inner sphere: E = 1 MPa, nu = 0.3
  // Outer sphere: E = 1 MPa, nu = 0.0 (or 0.3 in some cases)
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Inner sphere: K = 1/(3*(1-0.6)) = 0.833
  // Outer sphere: K = 1/(3*(1-0.0)) = 0.333
  mfem::Vector K_values({0.833, 0.333});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Inner sphere: G = 1/(2*(1+0.3)) = 0.385
  // Outer sphere: G = 1/(2*(1+0.0)) = 0.5
  mfem::Vector G_values({0.385, 0.5});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Apply symmetry boundary conditions
  // TODO: Constrain appropriate DOFs on x=0, y=0, z=0 planes

  // Apply compression via rigid plane displacement
  // Top plane moves down by uz = -10 mm over 10 seconds
  // TODO: This requires rigid body or constraint capabilities for the planes
  // For now, this could be approximated with prescribed displacement on outer sphere top

  const double total_compression = -10.0;
  auto compression = [total_compression](double t) -> mfem::Vector {
    mfem::Vector disp(3);
    disp = 0.0;
    disp(2) = total_compression * t / 10.0;  // Linear ramp over 10 seconds
    return disp;
  };
  // TODO: Apply to top plane or top of outer sphere

  // Add the contact interactions
  // Contact pair 1: Inner sphere / Outer sphere (inner surface)
  auto contact_interaction_1 = 0;
  std::set<int> inner_sphere_outer_surface({/* Attributes */});
  std::set<int> outer_sphere_inner_surface({/* Attributes */});
  solid_solver.addContactInteraction(contact_interaction_1, inner_sphere_outer_surface, outer_sphere_inner_surface,
                                     sphere_contact_options);

  // Contact pair 2: Outer sphere / Top plane (frictionless, single-pass)
  // TODO: Requires rigid plane implementation

  // Contact pair 3: Outer sphere / Bottom plane (frictionless, single-pass)
  // TODO: Requires rigid plane implementation

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  // 100 evenly spaced time steps over 10 seconds
  double dt = 0.1;

  for (int i{0}; i < 100; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
