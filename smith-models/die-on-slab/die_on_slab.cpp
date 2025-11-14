// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Die on Slab Contact Model
// Based on Example 2 from Puso & Laursen (2003)
// "A Mortar Segment-to-Segment Frictional Contact Method for Large Deformations"
// Reference: https://www.osti.gov/servlets/purl/15013715

#include <set>
#include <string>
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
  std::string name = "die_on_slab";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  // Construct the appropriate dimension mesh and give it to the data store
  // TODO: This would need a mesh file for the cylindrical die and slab
  // For now, using a placeholder mesh that would need to be generated
  std::string filename = SMITH_REPO_DIR "/data/meshes/ironing.mesh";
  std::shared_ptr<smith::Mesh> mesh = std::make_shared<smith::Mesh>(filename, "die_slab_mesh", 2, 0);

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = 50,
                                                  .print_level = 1};

  // Contact options
  // NOTE: Puso 2003 paper uses friction (mu=0.3), but Smith doesn't support friction yet
  // Using Frictionless as placeholder until friction is implemented
  smith::ContactOptions contact_options{.method = smith::ContactMethod::SingleMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 5.0e2,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh,
      {"bulk_mod", "shear_mod"});

  // Material properties from Puso 2003 paper
  // Slab: E = 1.0, nu = 0.3
  // Die (indenter): E = 1000.0, nu = 0.499
  // Convert to bulk and shear moduli: K = E/(3(1-2*nu)), G = E/(2(1+nu))

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  // Slab: K = 1.0/(3*(1-0.6)) = 0.833, Die: K = 1000/(3*(1-0.998)) = 166667
  mfem::Vector K_values({0.833, 166667.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  // Slab: G = 1.0/(2*(1+0.3)) = 0.385, Die: G = 1000/(2*(1+0.499)) = 333.6
  mfem::Vector G_values({0.385, 333.6});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Boundary conditions
  // Fix bottom of slab
  mesh->addDomainOfBoundaryElements("bottom_of_slab", smith::by_attr<dim>(5));
  solid_solver.setFixedBCs(mesh->domain("bottom_of_slab"));

  // Applied displacement to top of die
  mesh->addDomainOfBoundaryElements("top_of_die", smith::by_attr<dim>(12));
  auto applied_displacement = [](smith::tensor<double, dim>, double t) {
    constexpr double press_time = 0.2;  // Time to press down 1.4 units
    constexpr double press_depth = -1.4;
    constexpr double slide_distance = 4.0;
    constexpr double total_time = 1.5;

    smith::tensor<double, dim> u{};
    if (t <= press_time) {
      // Pressing phase: move down 1.4 units
      u[2] = press_depth * (t / press_time);
    } else {
      // Sliding phase: move 4 units horizontally
      double slide_time = t - press_time;
      double max_slide_time = total_time - press_time;
      u[0] = slide_distance * (slide_time / max_slide_time);
      u[2] = press_depth;
    }
    return u;
  };
  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("top_of_die"));

  // Add the contact interaction between slab and die surfaces
  auto contact_interaction_id = 0;
  std::set<int> slab_surface_attributes({6});
  std::set<int> die_surface_attributes({11});
  solid_solver.addContactInteraction(contact_interaction_id, slab_surface_attributes,
                                     die_surface_attributes, contact_options);

  // Finalize the data structures
  solid_solver.completeSetup();

  std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  double dt = 1.0;

  for (int i{0}; i < 26; ++i) {
    solid_solver.advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
