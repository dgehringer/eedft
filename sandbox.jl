using FFTW
using EEDFT
using BenchmarkTools

FFTW.set_num_threads(1)

fcc_al = Atoms([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0], [0.0 0.0 0.0; 0.5 0.5 0.0; 0.0 0.5 0.5; 0.5 0.0 0.5], species(:Ni, :Al, :Ni, :Ni))

# basis_c = PlaneWaveBasis(fcc_al, 100.0, ComplexF64)
basis = PlaneWaveBasis(fcc_al, 500.0, ComplexF64)

function make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, fft_func::Function, factor=100.0) where {T, TR}
    _, TG = fft_plan_data_types(basis.fft_plan_fw)
    r_data = rand(Tr, basis.r_grid_size) * Tr(factor)
    ggrid = fft_func(r_data)
    g_data = extract(ScalarFieldR{T, TR, TG}(basis, order, r_data), ggrid)
    ScalarFieldG{T, TR, TG}(basis, order, g_data)
end

make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, factor=100.0) where {T, TR <: Real} = make_scalar_field(Tr, basis, order, rfft, factor)
make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, factor=100.0) where {T, TR <: Complex} = make_scalar_field(Tr, basis, order, fft, factor)


density_roots  = [make_scalar_field(ComplexF64, basis, 1) for _ in 1:4]
densities = map((d¹²) -> d¹² * d¹², density_roots)

