module MPCPlotting

using PlotlyJS
using PlotlyJS: attr
using Printf

export save_mpc_plot

"""
save_mpc_plot(
    history_z,          # subsampled/block states
    history_z_times,    # times for block snapshots
    history_x,          # full micro-step states
    history_x_times,    # times for micro-step snapshots
    history_u,          # control inputs
    history_errors,     # norm(z - xT) at each block step
    XT,                 # target state
    max_micro_steps,
    nu,                 # smoothed block size (= nu_min + 1)
    Uref,               # smoothed reference input vector
    html_filename
)
"""
function save_mpc_plot(
    history_z,
    history_z_times,
    history_x,
    history_x_times,
    history_u,
    history_errors,
    XT,
    max_micro_steps,
    nu,
    Uref,               # smoothed Uref passed in for annotation
    html_filename
)
    n_z = length(history_z)
    n_x = length(history_x)
    n_u = length(history_u)
    n_e = length(history_errors)

    t_z = history_z_times[1:n_z]
    t_x = history_x_times[1:n_x]
    t_u = collect(1:n_u)

    p_sub  = [s[1] for s in history_z]
    v_sub  = [s[2] for s in history_z]
    p_full = [s[1] for s in history_x]
    v_full = [s[2] for s in history_x]

    log_err = log10.(history_errors[1:n_e] .+ 1e-16)

    # -- Uref horizontal lines on control panel ---------------
    uref_traces = PlotlyJS.GenericTrace[]
    colors = ["#e377c2", "#17becf", "#bcbd22", "#9467bd"]
    for j = 1:nu
        c = colors[mod1(j, length(colors))]
        tr = scatter(
            x    = [0, max_micro_steps],
            y    = [Uref[j], Uref[j]],
            mode = "lines",
            name = "Uref[$j] = $(@sprintf("%.4f", Uref[j]))",
            line = attr(dash="dot", width=1.5, color=c),
            yaxis = "y3"
        )
        push!(uref_traces, tr)
    end

    # -- State traces -----------------------------------------
    tr_p_full = scatter(
        x = t_x, y = p_full,
        mode = "lines",
        name = "position (micro-steps)",
        line = attr(width=2),
        yaxis = "y1"
    )

    tr_p_sub = scatter(
        x = t_z, y = p_sub,
        mode = "markers",
        name = "position (block snapshots)",
        marker = attr(size=8, symbol="diamond"),
        yaxis = "y1"
    )

    tr_p_target = scatter(
        x = [0, max_micro_steps],
        y = [XT[1], XT[1]],
        mode = "lines",
        name = "p_target",
        line = attr(dash="dash", width=2),
        yaxis = "y1"
    )

    tr_v_full = scatter(
        x = t_x, y = v_full,
        mode = "lines",
        name = "velocity (micro-steps)",
        line = attr(width=2),
        yaxis = "y2"
    )

    tr_v_sub = scatter(
        x = t_z, y = v_sub,
        mode = "markers",
        name = "velocity (block snapshots)",
        marker = attr(size=8, symbol="diamond"),
        yaxis = "y2"
    )

    tr_v_target = scatter(
        x = [0, max_micro_steps],
        y = [XT[2], XT[2]],
        mode = "lines",
        name = "v_target",
        line = attr(dash="dash", width=2),
        yaxis = "y2"
    )

    tr_u = scatter(
        x = t_u, y = history_u,
        mode = "lines",
        name = "u (micro)",
        line = attr(width=1.5),
        yaxis = "y3"
    )

    tr_logerr = scatter(
        x = t_z[1:n_e], y = log_err,
        mode = "lines+markers",
        name = "log10(norm(z - xT))",
        line = attr(width=2),
        marker = attr(size=6),
        yaxis = "y4"
    )

    # -- Build Uref annotation string for title ---------------
    uref_str = join(["Uref[$j]=$(@sprintf("%.3f",Uref[j]))" for j=1:nu], "  ")

    # -- Layout -----------------------------------------------
    layout = Layout(
        title = "Smoothed Uref MPC  (nu_min+1 = $nu)  |  $uref_str",
        xaxis  = attr(title="micro-steps", range=[0, max_micro_steps]),
        yaxis  = attr(domain=[0.76, 1.00], title="position"),
        yaxis2 = attr(domain=[0.52, 0.74], title="velocity"),
        yaxis3 = attr(domain=[0.26, 0.50], title="control u"),
        yaxis4 = attr(domain=[0.00, 0.24], title="log10(norm(z-xT))"),
        legend = attr(orientation="v", x=1.02, y=1.0),
        margin = attr(l=70, r=180, t=80, b=60)
    )

    all_traces = vcat(
        [tr_p_full, tr_p_sub, tr_p_target,
         tr_v_full, tr_v_sub, tr_v_target,
         tr_u],
        uref_traces,
        [tr_logerr]
    )

    # Ensure output directory exists
    mkpath(dirname(html_filename))

    fig = Plot(all_traces, layout)
    savefig(fig, html_filename)
    println("Plot saved -> $html_filename")
end

end # module MPCPlotting
