# run_mpc.jl
# Smoothed Uref strategy with nullity-2 (extra_steps = 2):
#   nu_smooth = nu_min + 2  ->  fat Bl  ->  2D null space
#   alpha* = [alpha1*, alpha2*] chosen via normal equations to minimise input gaps

include("src/Constants.jl")
include("src/SystemDynamics.jl")
include("src/BlockQP.jl")
include("src/MPCPlotting.jl")

using .Constants
using .SystemDynamics
using .BlockQP
using .MPCPlotting

using LinearAlgebra
using Printf
using JSON

# Save outputs into nu=4/output/ folder
const OUTPUT_DIR = joinpath(dirname(abspath(PROGRAM_FILE)), "output")

function run_mpc()

    # ----------------------------------------------------------
    #  Step 1: Find minimum reachability index nu_min
    # ----------------------------------------------------------
    nu_min      = compute_reachability_index(A, B, XT; max_nu = 10)
    extra_steps = 2                       # nullity = extra_steps
    nu          = nu_min + extra_steps    # smoothed block size
    Nblk        = max(1, div(N, nu))

    println("==================================================")
    @printf("  nu_min (reachability index) = %d\n", nu_min)
    @printf("  extra_steps (nullity)       = %d\n", extra_steps)
    @printf("  nu_smooth = nu_min + extra  = %d\n", nu)
    @printf("  Nblk (block horizon)        = %d\n", Nblk)
    println("==================================================")

    # ----------------------------------------------------------
    #  Step 2: Compute smoothed Uref ONCE offline
    #          Prints full pipeline breakdown exactly once
    # ----------------------------------------------------------
    smooth = compute_Uref_smoothed(A, B, XT, nu_min; extra_steps = extra_steps)
    Uref   = smooth.Uref

    println("==================================================")
    println("  Uref used throughout MPC loop:")
    for j = 1:nu
        @printf("    Uref[%d] = %12.6f\n", j, Uref[j])
    end
    println("==================================================\n")

    # ----------------------------------------------------------
    #  Step 3: Initialise state and history buffers
    # ----------------------------------------------------------
    z = copy(X0)

    history_z       = [copy(z)]
    history_z_times = [0]
    history_x       = [copy(z)]
    history_x_times = [0]
    history_u       = Float64[]
    history_errors  = Float64[]
    predicted_Js    = Float64[]
    micro_t         = 0

    # ----------------------------------------------------------
    #  Step 4: Initial predicted cost
    # ----------------------------------------------------------
    J_clarabel = Inf
    try
        sol_init   = solve_block_QP(z, XT, Nblk; A=A, B=B, Q=Q, R=R_scalar, nu=nu, Uref=Uref)
        J_clarabel = hasproperty(sol_init, :J) ? sol_init.J : Inf
    catch e
        @warn "Initial BlockQP solve failed: $e"
    end

    J_accumulated = 0.0

    # ----------------------------------------------------------
    #  Step 5: MPC loop -- Uref and nu fixed, no recomputation
    # ----------------------------------------------------------
    while micro_t < max_micro_steps

        sol = try
            solve_block_QP(z, XT, Nblk; A=A, B=B, Q=Q, R=R_scalar, nu=nu, Uref=Uref)
        catch e
            @warn "BlockQP failed at micro_t=$micro_t. Stopping. Error: $e"
            break
        end

        push!(predicted_Js, hasproperty(sol, :J) ? sol.J : Inf)

        # Apply first block (nu micro-steps)
        for j = 1:nu
            micro_t >= max_micro_steps && break

            u             = sol.U_sol[j, 1]
            control_cost  = R_scalar * (u - Uref[j])^2
            z             = apply_micro_step(A, B, z, u)
            state_cost    = dot(z - XT, Q * (z - XT))
            J_accumulated += state_cost + control_cost

            push!(history_u, u)
            micro_t += 1
            push!(history_x, copy(z))
            push!(history_x_times, micro_t)
        end

        push!(history_z, copy(z))
        push!(history_z_times, micro_t)
        push!(history_errors, norm(z - XT))
    end

    # ----------------------------------------------------------
    #  Step 6: Final results
    # ----------------------------------------------------------
    println("\n==================================================")
    println("  FINAL RESULTS")
    println("==================================================")
    @printf("  Final state            = [%.6f, %.6f]\n", z[1], z[2])
    @printf("  Final error norm(z-xT) = %.6e\n",         norm(z - XT))
    println("--------------------------------------------------")
    @printf("  J_clarabel  (initial predicted) = %.6f\n", J_clarabel)
    @printf("  J_accumulated (realized)        = %.6f\n", J_accumulated)
    @printf("  Number of MPC solves            = %d\n",   length(predicted_Js))
    println("--------------------------------------------------")
    @printf("  nu_smooth = %d  (nullity = %d)\n", nu, extra_steps)
    println("  Uref used (smoothed):")
    for j = 1:nu
        @printf("    Uref[%d] = %12.6f\n", j, Uref[j])
    end
    println("==================================================\n")

    # ----------------------------------------------------------
    #  Step 7: Save JSON into nu=4/output/
    # ----------------------------------------------------------
    mkpath(OUTPUT_DIR)
    out_json = joinpath(OUTPUT_DIR, json_filename)
    try
        summary = Dict(
            "nu_min"         => nu_min,
            "nu_smooth"      => nu,
            "nullity"        => extra_steps,
            "Uref"           => Uref,
            "final_state"    => [z[1], z[2]],
            "final_error"    => norm(z - XT),
            "J_clarabel"     => J_clarabel,
            "J_accumulated"  => J_accumulated,
            "predicted_Js"   => predicted_Js,
            "history_z"      => [[p[1], p[2]] for p in history_z],
            "history_x"      => [[p[1], p[2]] for p in history_x],
            "history_u"      => history_u,
            "history_errors" => history_errors
        )
        open(out_json, "w") do io
            JSON.print(io, summary, 4)
        end
        println("Saved summary -> $out_json")
    catch e
        @warn "Failed to save JSON: $e"
    end

    # ----------------------------------------------------------
    #  Step 8: Save plot into nu=4/output/
    # ----------------------------------------------------------
    out_html = joinpath(OUTPUT_DIR, html_filename)
    try
        save_mpc_plot(
            history_z, history_z_times,
            history_x, history_x_times,
            history_u, history_errors,
            XT, max_micro_steps, nu, Uref,
            out_html
        )
    catch e
        @warn "save_mpc_plot failed: $e"
    end

    return (
        final_state   = z,
        final_error   = norm(z - XT),
        J_clarabel    = J_clarabel,
        J_accumulated = J_accumulated,
        predicted_Js  = predicted_Js,
        Uref          = Uref
    )
end

run_mpc()
