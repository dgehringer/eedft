
using Test
using FFTW
using BenchmarkTools
import EEDFT: integrate, fft_plan_data_types, species, Atoms, PlaneWaveBasis, ScalarField, 𝔉⁻¹, 𝔉⁻¹!, 𝔉, 𝔉!, extract, fft_plan_data_types, is_G, is_r


fcc_al = Atoms([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], [0.0 0.0 0.0; 0.5 0.5 0.0; 0.0 0.5 0.5; 0.5 0.0 0.5], species(:Ni, :Al, :Ni, :Ni))


function make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, fft_func::Function) where {T, TR}
    _, TG = fft_plan_data_types(basis.fft_plan_fw)
    r_data = rand(Tr, basis.r_grid_size) * Tr(100.0)
    ggrid = fft_func(r_data)
    g_data = extract(ScalarField{T, TR, TG}(basis, order, nothing, r_data), ggrid)
    ScalarField{T, TR, TG}(basis, order, g_data, nothing)
end

make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int) where {T, TR <: Real} = make_scalar_field(Tr, basis, order, rfft)
make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int) where {T, TR <: Complex} = make_scalar_field(Tr, basis, order, fft)


@testset verbose=true "fourier transforms" begin

    atol = 1e-7

    @testset "FFT real-space type: $TR" for TR in (Float64, ComplexF64)

        basis = PlaneWaveBasis(fcc_al, 600.0, TR)

        @testset "G-shell: $shell" for shell in 1:max(keys(basis.G_shell_num_waves)...)
            
            
            f̃ = make_scalar_field(TR, basis, shell)
            f = 𝔉⁻¹(f̃)


            TR, TG = fft_plan_data_types(f̃.basis.fft_plan_fw)
            g_grid = zeros(TG, basis.G_grid_size)
            r_grid = zeros(TR, basis.r_grid_size)

            @testset "Fourier transforms" begin

                @assert f̃.r_data === nothing
                @test is_G(f̃)
                @test 𝔉(𝔉⁻¹(f̃)).g_data ≈ f̃.g_data atol=atol

                g_data_tmp = copy(f̃.g_data)
                
                𝔉⁻¹!(f̃)
                @test f̃.g_data === nothing
                𝔉!(f̃)
                @test is_r(f)
                @test f̃.g_data ≈ g_data_tmp atol=atol

                @test 𝔉⁻¹(𝔉(f)).r_data ≈ f.r_data atol=1e-5
            end

            # Check Parsevals theorem 1/N ∫f(𝐫)f*(𝐫)d𝐫 = ∫f̃(𝐆)f̃*(𝐆)d𝐆 
            @test integrate(f) ≈ integrate(𝔉(f)) rtol=1e-5

        end

    end

end