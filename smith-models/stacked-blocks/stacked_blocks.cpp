// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Stacked Blocks Contact Model
// Based on Figure 5 and Section 3.5 from Zimmerman & Ateshian (2018)
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
  std::string name = "stacked_blocks";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct four 2mm cube meshes with different refinements
  int num_refinements_block1{2};  // Top block (stiff)
  int num_refinements_block2{3};  // Second block (soft) - more refined
  int num_refinements_block3{2};  // Third block (stiff)
  int num_refinements_block4{3};  // Bottom block (soft) - more refined

  // Create Block 1 mesh (top, stiff)
  mfem::Mesh block1_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_block1; ++i) {
    block1_mesh.UniformRefinement();
  }
  block1_mesh.SetCurvature(p);
  block1_mesh.Transform([](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = 2.0 * x(0);  // Scale to 2mm
    p(1) = 2.0 * x(1);
    p(2) = 2.0 * x(2) + 6.0;  // Position at z=6 to z=8
  });

  // Create Block 2 mesh (second, soft)
  mfem::Mesh block2_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_block2; ++i) {
    block2_mesh.UniformRefinement();
  }
  block2_mesh.SetCurvature(p);
  block2_mesh.Transform([](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = 2.0 * x(0);
    p(1) = 2.0 * x(1);
    p(2) = 2.0 * x(2) + 4.0;  // Position at z=4 to z=6
  });

  // Create Block 3 mesh (third, stiff)
  mfem::Mesh block3_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_block3; ++i) {
    block3_mesh.UniformRefinement();
  }
  block3_mesh.SetCurvature(p);
  block3_mesh.Transform([](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = 2.0 * x(0);
    p(1) = 2.0 * x(1);
    p(2) = 2.0 * x(2) + 2.0;  // Position at z=2 to z=4
  });

  // Create Block 4 mesh (bottom, soft)
  mfem::Mesh block4_mesh{SMITH_REPO_DIR "/data/meshes/onehex.mesh"};
  for (int i{0}; i < num_refinements_block4; ++i) {
    block4_mesh.UniformRefinement();
  }
  block4_mesh.SetCurvature(p);
  block4_mesh.Transform([](const mfem::Vector& x, mfem::Vector& p) {
    p.SetSize(3);
    p(0) = 2.0 * x(0);
    p(1) = 2.0 * x(1);
    p(2) = 2.0 * x(2);  // Position at z=0 to z=2
  });

  std::vector<mfem::Mesh*> mesh_ptrs{&block1_mesh, &block2_mesh, &block3_mesh, &block4_mesh};
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh(mesh_ptrs.data(), static_cast<int>(mesh_ptrs.size())),
      "stacked_blocks_mesh", 0, 0);

  // Define boundary domains
  // Assuming standard cube attributes: 1=x-, 2=x+, 3=y-, 4=y+, 5=z-, 6=z+
  // Bottom of block 4 would be on attribute 5 of the 4th domain
  // Top of block 1 would be on attribute 6 of the 1st domain
  mesh->addDomainOfBoundaryElements("bottom_fixed", smith::by_attr<dim>(5));  // Bottom of Block 4
  mesh->addDomainOfBoundaryElements("top_load", smith::by_attr<dim>(6));      // Top of Block 1

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

  // Contact options for Interface 1 and 2 (low friction)
  smith::ContactOptions contact_low_friction{.method = smith::ContactMethod::SingleMortar,
                                             .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                             .type = smith::ContactType::Frictional,
                                             .penalty = 1.0,
                                             .friction_coeff = 0.05,
                                             .jacobian = smith::ContactJacobian::Exact};

  // Contact options for Interface 3 (high friction - sticking)
  smith::ContactOptions contact_high_friction{.method = smith::ContactMethod::SingleMortar,
                                              .enforcement = smith::ContactEnforcement::AugmentedLagrangian,
                                              .type = smith::ContactType::Frictional,
                                              .penalty = 10.0,
                                              .friction_coeff = 1.0,
                                              .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Zimmerman 2018 paper
  // Blocks 1 and 3: E = 10 MPa, nu = 0.1
  // Blocks 2 and 4: E = 0.3 MPa, nu = 0.4
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Block 1: K = 10/(3*(1-0.2)) = 4.167
  // Block 2: K = 0.3/(3*(1-0.8)) = 0.5
  // Block 3: K = 10/(3*(1-0.2)) = 4.167
  // Block 4: K = 0.3/(3*(1-0.8)) = 0.5
  mfem::Vector K_values({4.167, 0.5, 4.167, 0.5});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Block 1: G = 10/(2*(1+0.1)) = 4.545
  // Block 2: G = 0.3/(2*(1+0.4)) = 0.107
  // Block 3: G = 10/(2*(1+0.1)) = 4.545
  // Block 4: G = 0.3/(2*(1+0.4)) = 0.107
  mfem::Vector G_values({4.545, 0.107, 4.545, 0.107});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Fix bottom boundary of Block 4
  solid_solver.setFixedBCs(mesh->domain("bottom_fixed"));

  // Apply prescribed displacement to top of Block 1
  // uz = -2 mm over 10 time steps
  auto top_displacement = [](double t) -> mfem::Vector {
    mfem::Vector disp(3);
    disp = 0.0;
    disp(2) = -2.0 * t;  // -2 mm total displacement
    return disp;
  };
  solid_solver.setPrescribedDisplacement(mesh->domain("top_load"), top_displacement);

  // Add the contact interactions
  // Interface 1: Block 1 (bottom, attr 5) / Block 2 (top, attr 6)
  auto contact_interaction_1 = 0;
  std::set<int> block1_bottom({5});  // Bottom surface of Block 1
  std::set<int> block2_top({6});     // Top surface of Block 2
  solid_solver.addContactInteraction(contact_interaction_1, block1_bottom, block2_top, contact_low_friction);

  // Interface 2: Block 2 (bottom, attr 5) / Block 3 (top, attr 6)
  auto contact_interaction_2 = 1;
  std::set<int> block2_bottom({5});  // Bottom surface of Block 2
  std::set<int> block3_top({6});     // Top surface of Block 3
  solid_solver.addContactInteraction(contact_interaction_2, block2_bottom, block3_top, contact_low_friction);

  // Interface 3: Block 3 (bottom, attr 5) / Block 4 (top, attr 6)
  auto contact_interaction_3 = 2;
  std::set<int> block3_bottom({5});  // Bottom surface of Block 3
  std::set<int> block4_top({6});     // Top surface of Block 4
  solid_solver.addContactInteraction(contact_interaction_3, block3_bottom, block4_top, contact_high_friction);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 0.1;  // 10 time steps to reach uz = -2 mm

  for (int i{0}; i < 10; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
