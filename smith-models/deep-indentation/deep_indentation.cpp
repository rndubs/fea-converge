// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Deep Indentation of Square Blocks Contact Model
// Based on Figure 11 and Section 3.8 from Zimmerman & Ateshian (2018)
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
  std::string name = "deep_indentation";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Geometry parameters (all dimensions in mm)
  const double large_block_width = 1.5;
  const double large_block_depth = 1.5;
  const double large_block_height = 1.125;
  const double small_block_size = 0.5;  // Cube
  const double initial_gap = 0.0375;

  // Construct large block mesh (soft, bottom)
  int num_refinements_large{4};  // Needs good refinement for stress gradients
  mfem::Mesh large_block_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_large; ++i) {
    large_block_mesh.UniformRefinement();
  }
  large_block_mesh.SetCurvature(p);
  large_block_mesh.Transform([large_block_width, large_block_depth, large_block_height](
                                  const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = large_block_width * (x(0) - 0.5);   // Center at x=0
    p(1) = large_block_depth * (x(1) - 0.5);   // Center at y=0
    p(2) = large_block_height * x(2);          // Bottom at z=0
  });

  // Construct small block mesh (stiff indenter, top)
  // Paper notes coarse mesh in vertical direction is intentional
  int num_refinements_small{2};
  mfem::Mesh small_block_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_small; ++i) {
    small_block_mesh.UniformRefinement();
  }
  small_block_mesh.SetCurvature(p);
  small_block_mesh.Transform([small_block_size, large_block_height, initial_gap](
                                  const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = small_block_size * (x(0) - 0.5);   // Center at x=0
    p(1) = small_block_size * (x(1) - 0.5);   // Center at y=0
    p(2) = small_block_size * x(2) + large_block_height + initial_gap;  // Above large block
  });

  std::vector<mfem::Mesh*> mesh_ptrs{&large_block_mesh, &small_block_mesh};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
      "deep_indentation_mesh", 0, 0);

  // Define boundary domains
  // Large block: bottom surface (z=0) is attribute 5
  // Small block: top surface is attribute 6
  mesh->addDomainOfBoundaryElements("large_block_bottom", smith::by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("small_block_top", smith::by_attr<dim>(6));

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

  // Contact options - single pass with large block as primary surface
  // High penalty due to stiffness contrast
  // Gap tolerance convergence instead of penalty tolerance
  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                        .type = smith::ContactType::Frictional,
                                        .penalty = 150.0,
                                        .friction_coeff = 0.2,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Zimmerman 2018 paper
  // Large block: E = 1 MPa, nu = 0.3
  // Small block: E = 100 MPa, nu = 0.3
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Large block: K = 1/(3*(1-0.6)) = 0.833
  // Small block: K = 100/(3*(1-0.6)) = 83.333
  mfem::Vector K_values({0.833, 83.333});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Large block: G = 1/(2*(1+0.3)) = 0.385
  // Small block: G = 100/(2*(1+0.3)) = 38.462
  mfem::Vector G_values({0.385, 38.462});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Fix bottom of large block
  solid_solver.setFixedBCs(mesh->domain("large_block_bottom"));

  // Apply prescribed displacement to top of small block
  // uz = -0.6 mm over 1 second (100 time steps)
  // This indentation closes the 0.0375 mm gap and compresses 0.5625 mm into the large block
  const double total_displacement = -0.6;
  auto indenter_motion = [total_displacement](double t) -> mfem::Vector {
    mfem::Vector disp(3);
    disp = 0.0;
    disp(2) = total_displacement * t;  // Linear ramp
    return disp;
  };
  solid_solver.setPrescribedDisplacement(mesh->domain("small_block_top"), indenter_motion);

  // Add the contact interaction between small block and large block
  // Single-pass: large block surface is primary (noted in paper)
  auto contact_interaction_id = 0;
  std::set<int> small_block_surfaces({5});  // All exposed surfaces of small block
  std::set<int> large_block_surfaces({6});  // Top surface of large block
  // NOTE: As indentation progresses, side surfaces may also come into contact
  // This would require dynamic contact surface updates or broader surface definitions
  solid_solver.addContactInteraction(contact_interaction_id, large_block_surfaces, small_block_surfaces,
                                     contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  // 100 evenly spaced time steps over 1 second
  double dt = 0.01;

  for (int i{0}; i < 100; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
