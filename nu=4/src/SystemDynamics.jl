module SystemDynamics

using LinearAlgebra
using Printf

export build_block_B,
       compute_reachability_index,
       compute_Uref,
       compute_Uref_smoothed,
       compute_null_vector,
       compute_optimal_alpha,
       compute_optimal_alpha_multi,
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
#  Compute null vectors of Bl  (columns of N satisfy Bl*N = 0)
#  Returns all null vectors as columns of a matrix
# -------------------------------------------------------------
function compute_null_vector(Bl::AbstractMatrix)
    n, nu = size(Bl)
    @assert nu > n "Bl must be fat (nu > n). Got nu=$nu, n=$n."
    N = nullspace(Bl)
    @assert size(N, 2) >= 1 "Bl has trivial null space despite nu > n."
    return vec(N[:, 1])
end

# -------------------------------------------------------------
#  Compute optimal alpha* (scalar) for nullity-1 case
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
#  Compute optimal alpha* vector for multi-column null space
#
#  U(alpha) = U0 + V * alpha  where V is (nu x k) null basis
#  J(alpha) = sum_i (u_i - u_{i+1})^2
#
#  Gradient:  dJ/d(alpha) = 2 * D^T * D * (U0 + V*alpha)  = 0
#  where D is the (nu-1 x nu) finite-difference matrix
#
#  Normal equations:  (D*V)^T (D*V) alpha = -(D*V)^T (D*U0)
# -------------------------------------------------------------
function compute_optimal_alpha_multi(U0::Vector{Float64}, V::Matrix{Float64})
    nu, k = size(V)
    # Build finite-difference matrix D of size (nu-1) x nu
    D = zeros(nu - 1, nu)
    for i = 1:(nu - 1)
        D[i, i]   =  1.0
        D[i, i+1] = -1.0
    end
    DV  = D * V
    dU0 = D * U0
    # Solve normal equations
    alpha = -(DV' * DV) \ (DV' * dU0)
    return alpha
end

# -------------------------------------------------------------
#  Smoothed Uref pipeline -- supports arbitrary extra_steps
#
#  extra_steps = 1  ->  nullity = 1  (nu=3 case)
#  extra_steps = 2  ->  nullity = 2  (nu=4 case)
#
#  Steps:
#   1. nu = nu_min + extra_steps  ->  fat Bl, nullity = extra_steps
#   2. U0  = Bl^+ d               (minimum norm particular solution)
#   3. V   = null(Bl)             (null basis, nu x extra_steps matrix)
#   4. alpha* = argmin J(alpha)   (normal equations)
#   5. Uref = U0 + V * alpha*     (smoothed, still satisfies Bl Uref = d)
# -------------------------------------------------------------
function compute_Uref_smoothed(
    A::AbstractMatrix,
    B::AbstractMatrix,
    xT::Vector{Float64},
    nu_min::Int;
    extra_steps::Int = 1
)
    nu  = nu_min + extra_steps
    n   = size(A, 1)
    Anu = A^nu
    Bl  = build_block_B(A, B, nu)
    d   = (I(n) - Anu) * xT

    U0 = vec(pinv(Bl) * d)
    V  = nullspace(Bl)          # (nu x extra_steps) null basis
    @assert size(V, 2) >= extra_steps "Null space smaller than expected."

    V = V[:, 1:extra_steps]     # take exactly extra_steps null vectors

    if extra_steps == 1
        v     = vec(V)
        alpha_vec = [compute_optimal_alpha(U0, v)]
    else
        alpha_vec = compute_optimal_alpha_multi(U0, V)
    end

    Uref     = U0 .+ V * alpha_vec
    residual = norm(Bl * Uref - d)

    # -- Pretty-print full breakdown --------------------------
    println()
    println("+==================================================+")
    println("|  SMOOTHED Uref PIPELINE  (nu = $nu, nullity = $extra_steps)  |")
    println("+==================================================+")
    @printf("|  nu_min (reachability index)  = %d\n",   nu_min)
    @printf("|  extra_steps (nullity)        = %d\n",   extra_steps)
    @printf("|  nu_smooth = nu_min + extra   = %d\n",   nu)
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
    println("|  Null basis V  (Bl*V = 0), columns are null vectors")
    for j = 1:nu
        row = join([@sprintf("%10.6f", V[j, c]) for c in 1:extra_steps], "  ")
        @printf("|    V[%d, :] = %s\n", j, row)
    end
    println("+==================================================+")
    println("|  Optimal alpha* vector")
    for c = 1:extra_steps
        @printf("|    alpha[%d] = %12.6f\n", c, alpha_vec[c])
    end
    println("+==================================================+")
    println("|  *** FINAL Uref = U0 + V * alpha* ***")
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

    return (Uref = Uref, U0 = U0, V = V, alpha = alpha_vec, nu = nu, Bl = Bl)
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
