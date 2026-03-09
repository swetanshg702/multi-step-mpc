module Constants

using LinearAlgebra

export X0, XT, N, max_micro_steps, html_filename, json_filename,
       A, B, Q, R_scalar


const X0 = [1.0; 0.5]
const XT = [-0.5; 2.0]

const N = 20
const max_micro_steps = 200
const html_filename = "nu=4.html"
const json_filename  = "nu=4_results.json"

const A = [0.1 -0.5;
           0.2  0.8]

const B = reshape([1.0, 0.0], (2,1))

const Q = Diagonal([5.0, 5.0])
const R_scalar = 1.0

end # module Constants
