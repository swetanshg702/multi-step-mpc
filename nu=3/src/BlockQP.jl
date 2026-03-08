module BlockQP

using JuMP
using Clarabel
import MathOptInterface
const MOI = MathOptInterface
using LinearAlgebra

using Main.SystemDynamics: build_block_B

export solve_block_QP

# -------------------------------------------------------------
#  solve_block_QP
#
#  Uref and nu are computed ONCE offline in run_mpc and passed
#  in here directly -- no recomputation per iteration.
# -------------------------------------------------------------
function solve_block_QP(
    z0::Vector{Float64},
    xT::Vector{Float64},
    Nblk::Int;
    A::AbstractMatrix,
    B::AbstractMatrix,
    Q::AbstractMatrix,
    R::Real,
    nu::Int,                    # smoothed block size (= nu_min + 1)
    Uref::Vector{Float64}       # pre-computed smoothed Uref
)
    n  = size(A, 1)
    Anu = A^nu
    Bl = build_block_B(A, B, nu)

    # -- Build QP ---------------------------------------------
    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, U[1:nu, 1:Nblk])
    @variable(model, z[1:n, 1:Nblk+1])

    # Initial condition
    @constraint(model, z[:, 1] .== z0)

    # Block dynamics:  z_{k+1} = A^nu z_k + Bl U_k
    for k = 1:Nblk
        @constraint(model, z[:, k+1] .== Anu * z[:, k] + Bl * U[:, k])
    end

    # -- Cost -------------------------------------------------
    @expression(model, total_cost, 0.0)

    # Stage cost: state tracking + input deviation from Uref
    for k = 1:Nblk
        for i = 1:n
            total_cost += (z[i, k] - xT[i]) * Q[i, i] * (z[i, k] - xT[i])
        end
        for j = 1:nu
            total_cost += R * (U[j, k] - Uref[j])^2
        end
    end

    # Terminal cost  (Qf = 10 Q)
    Qf = 10.0 * Q
    for i = 1:n
        total_cost += (z[i, Nblk+1] - xT[i]) * Qf[i, i] * (z[i, Nblk+1] - xT[i])
    end

    @objective(model, Min, total_cost)
    optimize!(model)

    status = termination_status(model)
    if !(status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT))
        error("Block QP failed: $status")
    end

    U_sol = reshape([value(U[j, k]) for j in 1:nu, k in 1:Nblk], (nu, Nblk))
    z_sol = reshape([value(z[i, k]) for i in 1:n, k in 1:(Nblk+1)], (n, Nblk+1))
    Jobj  = objective_value(model)

    return (U_sol = U_sol, z_sol = z_sol, Uref = Uref, nu = nu, J = Jobj)
end

end # module BlockQP
