module SystemDynamics

using LinearAlgebra
using Printf

export build_block_B,
       compute_reachability_index,
       compute_Uref,
       compute_Uref_smoothed,
       compute_null_vector,
       compute_optimal_alpha,
       apply_micro_step

# -------------------------------------------------------------
#  Build lifted input matrix Bl in R^{n x nu}
#  Bl = [A^{nu-1}B, A^{nu-2}B, ..., B]
# -------------------------------------------------------------
function build_block_B(A::AbstractMatrix, B::AbstractMatrix, nu::Int)
    n = size(A, 1)
    Bl = zeros(n, nu)
    for i = 1:nu
        Bl[:, i] = A^(nu - i) * vec(B)
    end
    return Bl
end

# -------------------------------------------------------------
#  Find minimum nu (reachability index) such that Bl has full
#  row rank and (I - A^nu)xT is reachable
# -------------------------------------------------------------
function compute_reachability_index(
    A::AbstractMatrix,
    B::AbstractMatrix,
    xT::Vector{Float64};
    max_nu = 10,
    tol    = 1e-10
)
    n = size(A, 1)
    for nu = 1:max_nu
        Bl = build_block_B(A, B, nu)
        if rank(Bl) >= n
            Anu = A^nu
            d   = (I(n) - Anu) * xT
            TODO
            if norm(Bl * (pinv(Bl) * d) - d) < tol
                return nu
            end
        end
    end
    error("No reachability index found up to nu = $max_nu")
end

# -------------------------------------------------------------
#  Original Uref -- minimum norm pseudoinverse solution
#  Kept for backward compatibility
# -------------------------------------------------------------
function compute_Uref(
    A::AbstractMatrix,
    B::AbstractMatrix,
    xT::Vector{Float64},
    nu::Int
)
    Anu = A^nu
    Bl  = build_block_B(A, B, nu)
    d   = (I(size(A, 1)) - Anu) * xT
    return vec(pinv(Bl) * d)
end

# -------------------------------------------------------------
#  Compute null vector v of Bl  (Bl v = 0)
#  Requires fat Bl: nu > n
# -------------------------------------------------------------
function compute_null_vector(Bl::AbstractMatrix)
    n, nu = size(Bl)
    @assert nu > n "Bl must be fat (nu > n). Got nu=$nu, n=$n."
    N = nullspace(Bl)
    @assert size(N, 2) >= 1 "Bl has trivial null space despite nu > n."
    return vec(N[:, 1])
end

# -------------------------------------------------------------
#  Compute optimal alpha* that minimises consecutive input gaps
#
#  J(alpha) = sum_i (u_i - u_{i+1})^2   where U(alpha) = U0 + alpha*v
#
#  Closed form:  alpha* = -(sum_i delta_i * Delta_i) / (sum_i delta_i^2)
#  where Delta_i = U0[i]-U0[i+1],  delta_i = v[i]-v[i+1]
# -------------------------------------------------------------
function compute_optimal_alpha(U0::Vector{Float64}, v::Vector{Float64})
    nu = length(U0)
    numerator   = 0.0
    denominator = 0.0
    for i = 1:(nu - 1)
        Delta_i = U0[i] - U0[i+1]
        delta_i = v[i]  - v[i+1]
        numerator   += delta_i * Delta_i
        denominator += delta_i^2
    end
    abs(denominator) < 1e-14 && return 0.0
    return -numerator / denominator
end

# -------------------------------------------------------------
#  Smoothed Uref pipeline  --  uses nu_smooth = nu_min + 1
#
#  Steps:
#   1. nu_smooth = nu_min + 1  ->  fat Bl, nullity = 1
#   2. U0  = Bl^+ d            (minimum norm particular solution)
#   3. v   = null(Bl)          (null vector)
#   4. alpha* = argmin J(alpha) (closed form)
#   5. Uref = U0 + alpha* v    (smoothed, still satisfies Bl Uref = d)
# -------------------------------------------------------------
function compute_Uref_smoothed(
    A::AbstractMatrix,
    B::AbstractMatrix,
    xT::Vector{Float64},
    nu_min::Int
)
    nu  = nu_min + 1
    n   = size(A, 1)
    Anu = A^nu
    Bl  = build_block_B(A, B, nu)
    d   = (I(n) - Anu) * xT

    U0   = vec(pinv(Bl) * d)
    v    = compute_null_vector(Bl)
    alpha = compute_optimal_alpha(U0, v)
    Uref = U0 .+ alpha .* v

    residual = norm(Bl * Uref - d)

    # -- Pretty-print full breakdown --------------------------
    println()
    println("+==================================================+")
    println("|       SMOOTHED Uref PIPELINE  (nu = $nu)           |")
    println("+==================================================+")
    @printf("|  nu_min (reachability index)  = %d\n",   nu_min)
    @printf("|  nu_smooth = nu_min + 1       = %d\n",   nu)
    @printf("|  Bl shape                     = %d x %d\n", size(Bl)...)
    @printf("|  nullity(Bl) = nu - n         = %d\n",   nu - n)
    println("+==================================================+")
    println("|  d = (I - A^nu) xT")
    for i = 1:n
        @printf("|    d[%d] = %12.6f\n", i, d[i])
    end
    println("+==================================================+")
    println("|  Particular solution  U0 = Bl^+ d  (min-norm)")
    for j = 1:nu
        @printf("|    U0[%d] = %12.6f\n", j, U0[j])
    end
    println("+==================================================+")
    println("|  Null vector v  (Bl v = 0)")
    for j = 1:nu
        @printf("|    v[%d]  = %12.6f\n", j, v[j])
    end
    println("+==================================================+")
    @printf("|  Optimal alpha*  =  %12.6f\n", alpha)
    println("+==================================================+")
    println("|  *** FINAL Uref = U0 + alpha* v ***")
    for j = 1:nu
        @printf("|    Uref[%d] = %12.6f\n", j, Uref[j])
    end
    println("+==================================================+")
    println("|  Consecutive input gaps BEFORE smoothing (U0):")
    for i = 1:(nu-1)
        @printf("|    |u%d - u%d| = %12.6f\n", i, i+1, abs(U0[i] - U0[i+1]))
    end
    println("|  Consecutive input gaps AFTER  smoothing (Uref):")
    for i = 1:(nu-1)
        @printf("|    |u%d - u%d| = %12.6f\n", i, i+1, abs(Uref[i] - Uref[i+1]))
    end
    J_before = sum((U0[i]   - U0[i+1])^2   for i in 1:(nu-1))
    J_after  = sum((Uref[i] - Uref[i+1])^2 for i in 1:(nu-1))
    @printf("|  J(alpha=0)  = %12.6f  (pseudoinverse)\n", J_before)
    @printf("|  J(alpha*)   = %12.6f  (smoothed)\n",      J_after)
    @printf("|  Residual norm(Bl Uref - d) = %.2e  (must be ~0)\n", residual)
    println("+==================================================+")
    println()

    return (Uref = Uref, U0 = U0, v = v, alpha = alpha, nu = nu, Bl = Bl)
end

# -------------------------------------------------------------
#  Apply one micro-step:  x_{k+1} = A x_k + B u_k
# -------------------------------------------------------------
function apply_micro_step(
    A::AbstractMatrix,
    B::AbstractMatrix,
    x::Vector{Float64},
    u::Real
)
    return A * x .+ vec(B) * u
end

end # module SystemDynamics
