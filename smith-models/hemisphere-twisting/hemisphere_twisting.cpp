// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Hemisphere Twisting on Box Contact Model
// Based on Figure 7 and Section 3.6 from Zimmerman & Ateshian (2018)
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
  std::string name = "hemisphere_twisting";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Geometry parameters (L = 2 mm)
  const double L = 2.0;
  const double hemisphere_outer_radius = 1.0;
  const double hemisphere_inner_radius = 0.5;

  // Construct box mesh
  int num_refinements_box{3};
  mfem::Mesh box_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_box; ++i) {
    box_mesh.UniformRefinement();
  }
  box_mesh.SetCurvature(p);
  box_mesh.Transform([L](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = L * (x(0) - 0.5);  // Center at origin in x
    p(1) = L * (x(1) - 0.5);  // Center at origin in y
    p(2) = L * x(2);          // Box from z=0 to z=L
  });

  // Construct hollow hemisphere mesh
  // TODO: This requires a hollow hemisphere mesh with Ro=1.0, Ri=0.5
  // For now, using a scaled ball-nurbs mesh as placeholder
  int num_refinements_hemisphere{3};
  mfem::Mesh hemisphere_mesh{SMITH_REPO_DIR "/data/meshes/ball-nurbs.mesh"};
  for (int i{0}; i < num_refinements_hemisphere; ++i) {
    hemisphere_mesh.UniformRefinement();
  }
  hemisphere_mesh.SetCurvature(p);
  // Scale to outer radius and position above box
  hemisphere_mesh.Transform([hemisphere_outer_radius, L](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = hemisphere_outer_radius * x(0);
    p(1) = hemisphere_outer_radius * x(1);
    p(2) = hemisphere_outer_radius * x(2) + L;  // Position so bottom is at z=L
  });

  std::vector<mfem::Mesh*> mesh_ptrs{&box_mesh, &hemisphere_mesh};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
      "hemisphere_twisting_mesh", 0, 0);

  // Define boundary domains
  // Box: bottom surface (z=0) is attribute 5
  // Hemisphere: top surface is where we apply displacement/rotation
  mesh->addDomainOfBoundaryElements("box_bottom", smith::by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("hemisphere_top", smith::by_attr<dim>(6));

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

  // Contact options
  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                        .type = smith::ContactType::Frictional,
                                        .penalty = 10.0,
                                        .friction_coeff = 0.5,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Zimmerman 2018 paper
  // Box: E = 10 MPa, nu = 0.3
  // Hemisphere: E = 50 MPa, nu = 0.3
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Box: K = 10/(3*(1-0.6)) = 8.333
  // Hemisphere: K = 50/(3*(1-0.6)) = 41.667
  mfem::Vector K_values({8.333, 41.667});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Box: G = 10/(2*(1+0.3)) = 3.846
  // Hemisphere: G = 50/(2*(1+0.3)) = 19.231
  mfem::Vector G_values({3.846, 19.231});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Fix bottom of box
  solid_solver.setFixedBCs(mesh->domain("box_bottom"));

  // Apply prescribed displacement and rotation to top of hemisphere
  // Phase 1 (t=0 to 1): uz = -1 mm
  // Phase 2 (t=1 to 10): uz = -1 mm (held), Rz = pi rad
  const double indentation_depth = -1.0;
  const double total_rotation = M_PI;  // 180 degrees

  auto hemisphere_motion = [indentation_depth, total_rotation](double t) -> mfem::Vector {
    mfem::Vector disp(3);
    disp = 0.0;

    if (t <= 1.0) {
      // Indentation phase
      disp(2) = indentation_depth * t;  // Linear ramp to -1 mm
    } else {
      // Twisting phase - displacement held, rotation applied
      disp(2) = indentation_depth;  // Hold at -1 mm
      // NOTE: Rotation needs to be handled separately in Smith
      // This would require a rotation boundary condition
    }
    return disp;
  };
  solid_solver.setPrescribedDisplacement(mesh->domain("hemisphere_top"), hemisphere_motion);

  // TODO: Add prescribed rotation capability for twisting phase
  // This would require extending Smith to support rotational BCs
  // For now, this is noted in the model limitations

  // Add the contact interaction between hemisphere and box
  auto contact_interaction_id = 0;
  std::set<int> hemisphere_surface({/* Outer surface attributes */});
  std::set<int> box_top_surface({6});  // Top surface of box
  // NOTE: Actual surface attributes depend on mesh generation
  solid_solver.addContactInteraction(contact_interaction_id, hemisphere_surface, box_top_surface, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  // 100 time steps total: 10 for indentation (dt=0.1), 90 for twisting (dt=0.1)
  double dt = 0.1;

  for (int i{0}; i < 100; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
