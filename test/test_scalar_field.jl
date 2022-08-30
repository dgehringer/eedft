
using Test
using FFTW
using BenchmarkTools
#import EEDFT: integrate, fft_plan_data_types, species, Atoms, PlaneWaveBasis, ScalarField, 𝔉⁻¹, 𝔉⁻¹!, 𝔉, 𝔉!, extract, fft_plan_data_types, is_G, is_r
using EEDFT

fcc_al = Atoms([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], [0.0 0.0 0.0; 0.5 0.5 0.0; 0.0 0.5 0.5; 0.5 0.0 0.5], species(:Ni, :Al, :Ni, :Ni))


function make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, fft_func::Function, factor=100.0) where {T, TR}
    _, TG = fft_plan_data_types(basis.fft_plan_fw)
    r_data = rand(Tr, basis.r_grid_size) * Tr(factor)
    ggrid = fft_func(r_data)
    g_data = extract(ScalarFieldR{T, TR, TG}(basis, order, r_data), ggrid)
    ScalarFieldG{T, TR, TG}(basis, order, g_data)
end

make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, factor=100.0) where {T, TR <: Real} = make_scalar_field(Tr, basis, order, rfft, factor)
make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, factor=100.0) where {T, TR <: Complex} = make_scalar_field(Tr, basis, order, fft, factor)

scalar_field_types(::ScalarField{T, TR, TG}) where {T, TR, TG} = (T, TR, TG)

@testset verbose=true "fourier transforms" begin

    atol = 1e-6

    @testset "FFT real-space type: $TR" for TR in (Float64, ComplexF64)

        basis = PlaneWaveBasis(fcc_al, 300.0, TR)

        @testset "𝐆($shell)" for shell in 1:max(keys(basis.G_shell_num_waves)...)
            
            f̃ = make_scalar_field(TR, basis, shell)
            f = 𝔉⁻¹(f̃)

            T, _, __ = scalar_field_types(f)
            TR, TG = fft_plan_data_types(f̃.basis.fft_plan_fw)
            g_grid = zeros(TG, basis.G_grid_size)
            r_grid = zeros(TR, basis.r_grid_size)

            @testset "Fourier transforms - 𝔉 → 𝔉⁻¹" begin

                @test 𝔉(𝔉⁻¹(f̃)) ≈ f̃ atol=atol

                g_data_tmp = copy(f̃.g_data)
                
                f_tmp = 𝔉⁻¹!(f̃, r_grid)
                f̃_tmp = 𝔉!(f_tmp, g_grid)
                @test f̃_tmp.g_data ≈ g_data_tmp atol=atol

                @test 𝔉⁻¹(𝔉(f)) ≈ f atol=1e-5
            end

            # Check Parsevals theorem 1/N ∫f(𝐫)f*(𝐫)d𝐫 = ∫f̃(𝐆)f̃*(𝐆)d𝐆 
            @test integrate(f) ≈ integrate(𝔉(f)) rtol=1e-5

            @testset "-f(𝐫)" begin
                @test (-f).r_data ≈ -(f.r_data) atol=atol
                @test (-f̃).g_data ≈ -(f̃.g_data) atol=atol 
            end

            @testset "f(𝐫) + C" begin
                number = rand(TR) * 100
                result_r = ScalarFieldR{T, TR, TG}(f.basis, f.order, f.r_data .+ number)
                result_g = 𝔉(result_r)
                
                @test add_r(f, number) ≈ result_r atol=atol
                @test add_r(f̃, number) ≈ result_r atol=atol
                @test add_r(number, f) ≈ result_r atol=atol
                @test add_r(number, f̃) ≈ result_r atol=atol

                @test (number + f̃) ≈ (f + number) atol=atol
                @test (number +ᵣ f̃) ≈ (f +ᵣ number) atol=atol

                @test add_g(f, number) ≈ result_g atol=atol
                @test add_g(f̃, number) ≈ result_g atol=atol
                @test add_g(number, f) ≈ result_g atol=atol
                @test add_g(number, f̃) ≈ result_g atol=atol

            end

            @testset "f(𝐫) - C" begin
                number = rand(TR) * 100
                result_r = ScalarFieldR{T, TR, TG}(f.basis, f.order, f.r_data .- number)
                result_g = 𝔉(result_r)
                
                @test sub_r(f, number) ≈ result_r atol=atol
                @test sub_r(number, f) ≈ -result_r atol=atol
                @test sub_r(f̃, number) ≈ result_r atol=atol
                @test sub_r(number, f̃) ≈ -result_r atol=atol

                @test (number - f̃) ≈ -(f - number) atol=atol
                @test (number +ᵣ f̃) ≈ -(f +ᵣ number) atol=atol

                @test add_g(f, number) ≈ result_g atol=atol
                @test add_g(number, f) ≈ -result_g atol=atol
                @test add_g(f̃, number) ≈ result_g atol=atol
                @test add_g(number, f̃) ≈ -result_g atol=atol

            end

            @testset "𝐆($shell) → 𝐆($shell_other)" for shell_other in 1:max(keys(basis.G_shell_num_waves)...)
                
                g̃ = make_scalar_field(TR, basis, shell_other, 200.0)
                g = 𝔉⁻¹(g̃)

                @testset "f(𝐫) + g(𝐫)" begin
                    
                    result_r = ScalarFieldR{T, TR, TG}(f.basis, max(shell, shell_other), f.r_data .+ g.r_data) 
                    result_g = 𝔉(result_r)

                    # 𝐫 + 𝐫 → 𝐫
                    @test add_r(f, g) ≈ result_r atol=atol
                    # 𝐫 + 𝐆 → 𝐫
                    @test add_r(f, g̃) ≈ result_r atol=atol
                    # 𝐆 + 𝐫 → 𝐫
                    @test add_r(f̃, g) ≈ result_r atol=atol
                    # 𝐆 + 𝐆  → 𝐫
                    @test add_r(f̃, g̃) ≈ result_r atol=atol

                    @test add_r(f̃, g) ≈ add_r(f, g̃) atol=atol
                    @test g̃ +ᵣ f ≈ f̃ +ᵣ g atol=atol

                    # 𝐆 + 𝐆 → 𝐆
                    @test add_g(f̃, g̃) ≈ result_g atol=atol
                    # 𝐫 + 𝐆 → 𝐆
                    @test add_g(f, g̃) ≈ result_g atol=atol
                    # 𝐆 + 𝐫 → 𝐆
                    @test add_g(f̃, g) ≈ result_g atol=atol
                    # 𝐫 + 𝐫 → 𝐆
                    @test add_g(f, g) ≈ result_g atol=atol

                    @test add_g(f̃, g) ≈ add_g(f, g̃) atol=atol
                    @test g̃ + f ≈ f̃ + g atol=atol
                end

                @testset "f(𝐫) - g(𝐫)" begin
                    result_r = ScalarFieldR{T, TR, TG}(f.basis, max(shell, shell_other), f.r_data .- g.r_data)
                    result_g = 𝔉(result_r)
                    # print("SUM: $(sum(result_r)) \n")

                    # 𝐫 + 𝐫 → 𝐫
                    @test sub_r(f, g) ≈ result_r atol=atol
                    @test sub_r(g, f) ≈ -result_r atol=atol

                    # 𝐫 + 𝐆 → 𝐫
                    @test sub_r(f, g̃) ≈ result_r atol=atol
                    @test sub_r(g̃, f) ≈ -result_r atol=atol

                    # 𝐆 + 𝐫 → 𝐫
                    @test sub_r(f̃, g) ≈ result_r atol=atol
                    @test sub_r(g, f̃) ≈ -result_r atol=atol
                    
                    # 𝐆 + 𝐆 → 𝐆
                    @test sub_g(f̃, g̃) ≈ result_g atol=atol
                    @test sub_g(g̃, f̃) ≈ -result_g atol=atol

                    # 𝐆 + 𝐆  → 𝐫
                    @test sub_r(f̃, g̃) ≈ result_r atol=atol
                    @test sub_r(g̃, f̃) ≈ -result_r atol=atol

                    # 𝐫 + 𝐆 → 𝐆
                    @test sub_g(f, g̃) ≈ result_g atol=atol
                    @test sub_g(g̃, f) ≈ -result_g atol=atol

                    # 𝐆 + 𝐫 → 𝐆
                    @test sub_g(f̃, g) ≈ result_g atol=atol
                    @test sub_g(g, f̃) ≈ -result_g atol=atol

                    # 𝐫 + 𝐫 → 𝐆
                    @test sub_g(f, g) ≈ result_g atol=atol
                    @test sub_g(g, f) ≈ -result_g atol=atol
                end
            end
        end
    end
end