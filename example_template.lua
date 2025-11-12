-- Smith FEA Input Deck Template
-- This template uses placeholder syntax for parameter optimization
-- Replace {{ parameter_name }} with actual values during optimization

-- Mesh configuration
mesh = {
    file = "mesh.e",
    serial_refinement = 0,
    parallel_refinement = 0
}

-- Solver configuration
solver = {
    -- Linear solver settings
    linear_solver = {
        type = "iterative",
        max_iterations = {{ max_iterations }},  -- ML parameter
        relative_tolerance = {{ solver_tolerance }},  -- ML parameter
        absolute_tolerance = {{ solver_tolerance }} * 1e-2,
        preconditioner = "AMG",
        print_level = 1,
        
        -- Multigrid settings
        amg = {
            coarsen_type = 10,
            agg_levels = 1,
            relax_type = 6,
            theta = 0.25,
            interpolation_type = 6,
            p_max = 4
        }
    },
    
    -- Nonlinear solver settings
    nonlinear_solver = {
        relative_tolerance = {{ solver_tolerance }} * 10,
        absolute_tolerance = {{ solver_tolerance }},
        max_iterations = 50,
        print_level = 1
    },
    
    -- Time integration
    time_stepping = {
        start_time = 0.0,
        end_time = 1.0,
        dt = 0.01,
        adaptive = true,
        min_dt = 1e-6,
        max_dt = 0.1
    }
}

-- Material properties
materials = {
    {
        name = "steel",
        density = 7850.0,  -- kg/m^3
        
        elastic = {
            youngs_modulus = 200e9,  -- Pa
            poissons_ratio = 0.3
        },
        
        thermal = {
            conductivity = 50.0,  -- W/(m*K)
            specific_heat = 500.0  -- J/(kg*K)
        }
    }
}

-- Contact mechanics (if using Tribol)
contact = {
    enabled = true,
    
    penalty_method = {
        penalty_parameter = {{ penalty_parameter }},  -- ML parameter
        penalty_coefficient = {{ penalty_coefficient }},  -- ML parameter
        contact_tolerance = {{ solver_tolerance }} * 100,
    },
    
    friction = {
        enabled = false,
        coefficient = 0.3
    }
}

-- Boundary conditions
boundary_conditions = {
    -- Displacement BC
    {
        type = "displacement",
        boundary = "left_surface",
        component = "x",
        value = 0.0
    },
    
    -- Load BC
    {
        type = "traction",
        boundary = "right_surface",
        component = "x",
        value = 1e6  -- Pa
    },
    
    -- Temperature BC
    {
        type = "temperature",
        boundary = "bottom_surface",
        value = 300.0  -- K
    }
}

-- Output configuration
output = {
    type = "paraview",
    file_prefix = "results/smith_output",
    time_step_interval = 10,
    
    fields = {
        "displacement",
        "stress",
        "strain",
        "temperature",
        "contact_pressure"
    }
}

-- Performance monitoring
monitoring = {
    print_residuals = true,
    print_timing = true,
    print_memory = false
}
