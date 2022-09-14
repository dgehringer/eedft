

import FFTW
using IterTools


function build_fft_plans(T::Type, planer::Function, fft_size::Tuple{Int, Int, Int}; flags=FFTW.MEASURE)
    out_fft_plan = planer(Array{T, 3}(undef, fft_size...), flags=flags)
    @info("[FFT]: Planing $(T <: Real ? "R" : "")FFT for type $(T): $(out_fft_plan.sz) → $(out_fft_plan.osz)")
    out_fft_plan, inv(out_fft_plan).p, out_fft_plan.sz, out_fft_plan.osz
end

build_fft_plans(T::Type{V}, fft_size::Tuple{Int, Int, Int}; flags=FFTW.MEASURE) where {V <: Real} = build_fft_plans(V, FFTW.plan_rfft, fft_size; flags=flags)
build_fft_plans(T::Type{V}, fft_size::Tuple{Int, Int, Int}; flags=FFTW.MEASURE) where {V <: Complex} = build_fft_plans(V, FFTW.plan_fft, fft_size; flags=flags)

fft_plan_input_data_type(::FFTW.cFFTWPlan{T, D, I, R}) where {T, D, I, R} = T
fft_plan_input_data_type(::FFTW.rFFTWPlan{T, D, I, R}) where {T, D, I, R} = T

fft_plan_data_types(p::FFTW.FFTWPlan) = (fft_plan_input_data_type(p), fft_plan_input_data_type(inv(p).p))


# returns the lengths of the bounding rectangle in reciprocal space
# that encloses the sphere of radius Gmax
function bounding_rectangle(lattice::AbstractMatrix{T}, Gmax; tol=sqrt(eps(T))) where {T <: Real}
    # If |B G| ≤ Gmax, then
    # |Gi| = |e_i^T B^-1 B G| ≤ |B^-T e_i| Gmax = |A_i| Gmax
    # with B the reciprocal lattice matrix, e_i the i-th canonical
    # basis vector and A_i the i-th column of the lattice matrix
    Glims = [norm(lattice[:, i]) / 2T(π) * Gmax for i in 1:3]

    # Round up, unless exactly zero (in which case keep it zero in
    # order to just have one G vector for 1D or 2D systems)
    Glims = [Glim == 0 ? 0 : ceil(Int, Glim .- tol) for Glim in Glims]
    Glims
end

function compute_Glims(lattice::Mat3{T}, Ecut::T; supersampling=2, tol=sqrt(eps(T))) where {T}
    Gmax = supersampling * sqrt(2*Ecut)
    bounding_rectangle(lattice, Gmax; tol=tol)
end

function compute_fft_size(lattice::Mat3{T}, Ecut::T; ensure_smallprimes=true, supersampling=2) where T
    Glims = compute_Glims(lattice::Mat3{T}, Ecut; supersampling=supersampling)
    fft_size = Vec3(2.0 .* Glims .+ 1)
    if ensure_smallprimes
        fft_size = nextprod.(Ref([2, 3, 5]), fft_size)
    end
    Tuple{Int, Int, Int}(fft_size)
end

function G_vectors(T::Type, G_grid_size::Tuple{Int, Int, Int}, frequenies::Tuple{Function, Function, Function})
    frequenies = (convert(Vector{Int}, freq_func(n, n)) for (freq_func, n) in zip(frequenies, G_grid_size))
    (Vec3{Int}(i,j,k) for (i, j, k) in product(frequenies...))
end

G_vectors(T::Type{V}, G_grid_size::Tuple{Int, Int, Int}) where {V <: Real} = G_vectors(V, G_grid_size, (FFTW.rfftfreq, FFTW.fftfreq, FFTW.fftfreq))
G_vectors(T::Type{V}, G_grid_size::Tuple{Int, Int, Int}) where {V <: Complex} = G_vectors(V, G_grid_size, (FFTW.fftfreq, FFTW.fftfreq, FFTW.fftfreq))

function G_vectors_with_indices(T::Type, G_grid_size::Tuple{Int, Int, Int}, frequenies::Tuple{Function, Function, Function})
    a, b, c = (convert(Vector{Int}, freq_func(n, n)) for (freq_func, n) in zip(frequenies, G_grid_size))
    ( (CartesianIndex(ii, jj, kk), Vec3{Int}(i,j,k)) for (ii, i) in enumerate(a), (jj, j) in enumerate(b), (kk, k) in enumerate(c))
end

G_vectors_with_indices(T::Type{V}, G_grid_size::Tuple{Int, Int, Int}) where {V <: Real} = G_vectors_with_indices(T, G_grid_size, (FFTW.rfftfreq, FFTW.fftfreq, FFTW.fftfreq))
G_vectors_with_indices(T::Type{V}, G_grid_size::Tuple{Int, Int, Int}) where {V <: Complex} = G_vectors_with_indices(T, G_grid_size, (FFTW.fftfreq, FFTW.fftfreq, FFTW.fftfreq))


function r_vectors(T::Type, r_grid_size::Tuple{Int, Int, Int}) 
    N1, N2, N3 = r_grid_size
    (Vec3{T}(T(i-1) / N1, T(j-1) / N2, T(k-1) / N3) for i = 1:N1, j = 1:N2, k = 1:N3)
end


function G_to_r!(f_fourier::Array{T, 3}, bw_plan, f_real::Array{V, 3}) where {T, V}
    @assert size(f_fourier) == bw_plan.sz
    @assert size(f_real) == bw_plan.osz;
    mul!(f_real, bw_plan, f_fourier)
    f_real
end

function G_to_r(f_fourier::Array{T, 3}, bw_plan) where {T}
    @assert size(f_fourier) == bw_plan.sz
    bw_plan * f_fourier
end


function r_to_G!(f_real::Array{T, 3}, fw_plan, f_fourier::Array{V, 3}) where {T, V}
    @assert size(f_real) == fw_plan.sz
    @assert size(f_fourier) == fw_plan.osz;
    mul!(f_fourier, fw_plan, f_real)
    f_fourier
end

function r_to_G(f_real::Array{T, 3}, fw_plan) where {T}
    @assert size(f_real) == fw_plan.sz
    fw_plan * f_real
end
