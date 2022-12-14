
using Test
using FFTW
using BenchmarkTools
#import EEDFT: integrate, fft_plan_data_types, species, Atoms, PlaneWaveBasis, ScalarField, ๐โปยน, ๐โปยน!, ๐, ๐!, extract, fft_plan_data_types, is_G, is_r
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

    @testset "FFT real-space type: $TR" for TR in (ComplexF64, )

        basis = PlaneWaveBasis(fcc_al, 100.0, TR)
        max_shells = max(keys(basis.G_shell_num_waves)...)

        @testset "๐($shell)" for shell in 1:max_shells
            
            fฬ = make_scalar_field(TR, basis, shell)
            f = ๐โปยน(fฬ)

            T, _, __ = scalar_field_types(f)
            TR, TG = fft_plan_data_types(fฬ.basis.fft_plan_fw)
            g_grid = zeros(TG, basis.G_grid_size)
            r_grid = zeros(TR, basis.r_grid_size)

            @testset "Fourier transforms - ๐ โ ๐โปยน" begin

                @test ๐(๐โปยน(fฬ)) โ fฬ atol=atol

                g_data_tmp = copy(fฬ.g_data)
                
                f_tmp = ๐โปยน!(fฬ, r_grid)
                fฬ_tmp = ๐!(f_tmp, g_grid)
                @test fฬ_tmp.g_data โ g_data_tmp atol=atol

                @test ๐(๐โปยน(fฬ)) โ fฬ
                @test ๐โปยน(๐(f)) โ f

                @test ๐(๐โปยน(๐(๐โปยน(fฬ)))) โ fฬ
                @test ๐โปยน(๐(๐โปยน(๐(f)))) โ f
            end

            # Check Parsevals theorem 1/N โซf(๐ซ)f*(๐ซ)d๐ซ = โซfฬ(๐)fฬ*(๐)d๐ 
            @test integrate(f) โ integrate(๐(f)) rtol=1e-5

            @testset "-f(๐ซ)|fฬ(๐)" begin
                @test (-f).r_data โ -(f.r_data) atol=atol
                @test (-fฬ).g_data โ -(fฬ.g_data) atol=atol 
            end

            @testset "f(๐ซ)|fฬ(๐) + C" begin
                number = rand(TR) * 100
                result_r = ScalarFieldR{T, TR, TG}(f.basis, f.order, f.r_data .+ number)
                result_g = ๐(result_r)
                
                @test add_r(f, number) โ result_r atol=atol
                @test add_r(fฬ, number) โ result_r atol=atol
                @test add_r(number, f) โ result_r atol=atol
                @test add_r(number, fฬ) โ result_r atol=atol

                @test (number + fฬ) โ (f + number) atol=atol
                @test (number +แตฃ fฬ) โ (f +แตฃ number) atol=atol

                @test add_g(f, number) โ result_g atol=atol
                @test add_g(fฬ, number) โ result_g atol=atol
                @test add_g(number, f) โ result_g atol=atol
                @test add_g(number, fฬ) โ result_g atol=atol
            end

            @testset "f(๐ซ)|fฬ(๐) - C" begin
                number = rand(TR) * 100
                result_r = ScalarFieldR{T, TR, TG}(f.basis, f.order, f.r_data .- number)
                result_g = ๐(result_r)
                
                @test sub_r(f, number) โ result_r atol=atol
                @test sub_r(number, f) โ -result_r atol=atol
                @test sub_r(fฬ, number) โ result_r atol=atol
                @test sub_r(number, fฬ) โ -result_r atol=atol

                @test (number - fฬ) โ -(f - number) atol=atol
                @test (number +แตฃ fฬ) โ -(f +แตฃ number) atol=atol

                @test sub_g(f, number) โ result_g atol=atol
                @test sub_g(number, f) โ -result_g atol=atol
                @test sub_g(fฬ, number) โ result_g atol=atol
                @test sub_g(number, fฬ) โ -result_g atol=atol
            end

            @testset "inv(f(๐ซ)|fฬ(๐))" begin
                result_r = ScalarFieldR{T, TR, TG}(f.basis, f.order, inv.(f.r_data))
                result_g = ๐(result_r)

                @test inv_r(f) โ result_r atol=atol
                @test inv_r(fฬ) โ result_r atol=atol

                @test inv_g(f) โ result_g atol=atol
                @test inv_g(fฬ) โ result_g atol=atol
            end

            @testset "C * f(๐ซ)|fฬ(๐) * C" begin
                number = rand(TR) * 100
                result_r = ScalarFieldR{T, TR, TG}(f.basis, f.order, f.r_data .* number)
                result_g = ๐(result_r)

                @test mul_r(f, number) โ result_r atol=atol
                @test mul_r(number, fฬ) โ result_r atol=atol
                @test mul_r(number, f) โ mul_r(fฬ, number) atol=atol
                @test (number *แตฃ f) โ (fฬ *แตฃ number) atol=atol
                
                @test mul_g(f, number) โ result_g atol=atol
                @test mul_g(number, fฬ) โ result_g atol=atol
                @test mul_g(number, f) โ mul_g(fฬ, number) atol=atol
                @test (number * f) โ (fฬ * number) atol=atol
            end

            @testset "ฮf|ฮfฬ" begin
                @test ฮ(f) โ diff_g(f, 2, 1) + diff_g(f, 2, 2) + diff_g(f, 2, 3)
                @test ฮ(fฬ) โ diff_g(fฬ, 2, 1) + diff_g(fฬ, 2, 2) + diff_g(fฬ, 2, 3)

                @test ฮแตฃ(f) โ diff_r(f, 2, 1) + diff_r(f, 2, 2) + diff_r(f, 2, 3)
                @test ฮแตฃ(fฬ) โ diff_r(fฬ, 2, 1) + diff_r(fฬ, 2, 2) + diff_r(fฬ, 2, 3)
            end

            @testset "โแตขf, โโฑผf, โโf" begin
                for (ax, โโ, โโสณ) in zip(1:3, (โแตข, โโฑผ, โโ),  (โแตขสณ, โโฑผสณ, โโสณ))
                    Gแตข = (inforder(f) ? basis.๐[ax, :, :, :] : f.basis.G_shell_vectors[f.order][ax, :]) * 1im
                    result_g = (fฬ.g_data .* Gแตข)
                    

                    @test โโ(๐(f)) โ โโ(fฬ)
                    @test โโ(f) โ โโ(๐โปยน(fฬ))
                    @test โโ(fฬ).g_data โ result_g
                    # @test โโ(f).g_data ./ 9.223372036854778e+18 โ result_g
                    
                    
                    @test โโ(๐(f)) โ โโ(fฬ)
                    @test f โ ๐โปยน(fฬ)

                    # @test ๐(โโสณ(fฬ)).g_data ./ 9.223372036854778e+18 โ result_g
                    # @test ๐(โโสณ(f)).g_data ./ (9.223372036854778e+18^2) โ result_g

                    # Test Satz von Schwarz โโโโ = โโโโ
                    for (sax, โโ, โโสณ) in zip(1:3, (โแตข, โโฑผ, โโ),  (โแตขสณ, โโฑผสณ, โโสณ))
                        sax == ax && continue

                        @test โโ(โโ(f)) โ โโ(โโ(f))
                        @test โโ(โโ(fฬ)) โ โโ(โโ(f))
                        @test โโ(โโ(f)) โ โโ(โโ(fฬ))
                        @test โโ(โโ(fฬ)) โ โโ(โโ(fฬ))

                        @test โโ(โโสณ(fฬ)) โ โโ(โโ(fฬ))
                        @test โโสณ(โโ(fฬ)) โ โโสณ(โโ(fฬ))
                        @test โโสณ(โโสณ(fฬ)) โ โโสณ(โโสณ(fฬ))
                        @test โโ(โโสณ(fฬ)) โ โโ(โโสณ(fฬ))
                    end
                end
            end

            @testset "C / f(๐ซ)|fฬ(๐) / C" begin
                number = rand(TR) * 100
                result_r1 = ScalarFieldR{T, TR, TG}(f.basis, f.order, inv(number) .* f.r_data)
                result_r2 = ScalarFieldR{T, TR, TG}(f.basis, f.order, number .* inv.(f.r_data))
                result_g1 = ๐(result_r1)
                result_g2 = ๐(result_r2)

                @test div_r(number, f) โ result_r1 atol=atol
                @test div_r(number, fฬ) โ result_r1 atol=atol
                @test div_r(f, number) โ result_r2 atol=atol
                @test div_r(fฬ, number) โ result_r2 atol=atol
                @test number /แตฃ f โ result_r1 atol=atol
                @test fฬ /แตฃ number โ result_r2 atol=atol

                @test div_g(number, f) โ result_g1 atol=atol
                @test div_g(number, fฬ) โ result_g1 atol=atol
                @test div_g(f, number) โ result_g2 atol=atol
                @test div_g(fฬ, number) โ result_g2 atol=atol
                @test number / f โ result_g1 atol=atol
                @test fฬ / number โ result_g2 atol=atol
            end

            @testset "๐($shell) โ ๐($shell_other)" for shell_other in 1:max_shells
                
                gฬ = make_scalar_field(TR, basis, shell_other, 200.0)
                g = ๐โปยน(gฬ)

                @testset "f(๐ซ)|fฬ(๐) + g(๐ซ)|gฬ(๐)" begin
                    
                    result_r = ScalarFieldR{T, TR, TG}(f.basis, max(shell, shell_other), f.r_data .+ g.r_data) 
                    result_g = ๐(result_r)

                    # ๐ซ + ๐ซ โ ๐ซ
                    @test add_r(f, g) โ result_r atol=atol
                    # ๐ซ + ๐ โ ๐ซ
                    @test add_r(f, gฬ) โ result_r atol=atol
                    # ๐ + ๐ซ โ ๐ซ
                    @test add_r(fฬ, g) โ result_r atol=atol
                    # ๐ + ๐  โ ๐ซ
                    @test add_r(fฬ, gฬ) โ result_r atol=atol

                    @test add_r(fฬ, g) โ add_r(f, gฬ) atol=atol
                    @test gฬ +แตฃ f โ fฬ +แตฃ g atol=atol

                    # ๐ + ๐ โ ๐
                    @test add_g(fฬ, gฬ) โ result_g atol=atol
                    # ๐ซ + ๐ โ ๐
                    @test add_g(f, gฬ) โ result_g atol=atol
                    # ๐ + ๐ซ โ ๐
                    @test add_g(fฬ, g) โ result_g atol=atol
                    # ๐ซ + ๐ซ โ ๐
                    @test add_g(f, g) โ result_g atol=atol

                    @test add_g(fฬ, g) โ add_g(f, gฬ) atol=atol
                    @test gฬ + f โ fฬ + g atol=atol
                end

                @testset "f(๐ซ)|fฬ(๐) - g(๐ซ)|gฬ(๐)" begin
                    result_r = ScalarFieldR{T, TR, TG}(f.basis, max(shell, shell_other), f.r_data .- g.r_data)
                    result_g = ๐(result_r)
                    # print("SUM: $(sum(result_r)) \n")

                    # ๐ซ - ๐ซ โ ๐ซ
                    @test sub_r(f, g) โ result_r atol=atol
                    @test sub_r(g, f) โ -result_r atol=atol

                    # ๐ซ - ๐ โ ๐ซ
                    @test sub_r(f, gฬ) โ result_r atol=atol
                    @test sub_r(gฬ, f) โ -result_r atol=atol

                    # ๐ - ๐ซ โ ๐ซ
                    @test sub_r(fฬ, g) โ result_r atol=atol
                    @test sub_r(g, fฬ) โ -result_r atol=atol
                    
                    # ๐ - ๐ โ ๐
                    @test sub_g(fฬ, gฬ) โ result_g atol=atol
                    @test sub_g(gฬ, fฬ) โ -result_g atol=atol

                    # ๐ - ๐  โ ๐ซ
                    @test sub_r(fฬ, gฬ) โ result_r atol=atol
                    @test sub_r(gฬ, fฬ) โ -result_r atol=atol

                    # ๐ซ - ๐ โ ๐
                    @test sub_g(f, gฬ) โ result_g atol=atol
                    @test sub_g(gฬ, f) โ -result_g atol=atol

                    # ๐ - ๐ซ โ ๐
                    @test sub_g(fฬ, g) โ result_g atol=atol
                    @test sub_g(g, fฬ) โ -result_g atol=atol

                    # ๐ซ - ๐ซ โ ๐
                    @test sub_g(f, g) โ result_g atol=atol
                    @test sub_g(g, f) โ -result_g atol=atol
                    
                    @test gฬ - f โ -(fฬ - g) atol=atol
                    @test gฬ -แตฃ f โ -(fฬ -แตฃ g) atol=atol
                end

                @testset "f(๐ซ)|fฬ(๐) * g(๐ซ)|gฬ(๐)" begin
                    shell_result = shell + shell_other > max_shells ? InfOrder : shell + shell_other
                    result_r = ScalarFieldR{T, TR, TG}(f.basis, shell_result, f.r_data .* g.r_data)
                    result_g = ๐(result_r)
    
                    # ๐ซ + ๐ซ โ ๐ซ
                    @test mul_r(f, g) โ result_r atol=atol
                    @test mul_r(g, f) โ result_r atol=atol
                    # ๐ซ + ๐ โ ๐ซ
                    @test mul_r(f, gฬ) โ result_r atol=atol
                    @test mul_r(gฬ, f) โ result_r atol=atol
                    # ๐ + ๐ซ โ ๐ซ
                    @test mul_r(fฬ, g) โ result_r atol=atol
                    @test mul_r(g, fฬ) โ result_r atol=atol
                    # ๐ + ๐  โ ๐ซ
                    @test mul_r(fฬ, gฬ) โ result_r atol=atol
                    @test mul_r(gฬ, fฬ) โ result_r atol=atol
    
                    @test mul_r(fฬ, g) โ mul_r(f, gฬ) atol=atol
                    @test gฬ *แตฃ f โ fฬ *แตฃ g atol=atol

                    # ๐ - ๐ โ ๐
                    @test mul_g(fฬ, gฬ) โ result_g atol=atol
                    @test mul_g(gฬ, fฬ) โ result_g atol=atol

                    # ๐ซ - ๐ โ ๐
                    @test mul_g(f, gฬ) โ result_g atol=atol
                    @test mul_g(gฬ, f) โ result_g atol=atol

                    # ๐ - ๐ซ โ ๐
                    @test mul_g(fฬ, g) โ result_g atol=atol
                    @test mul_g(g, fฬ) โ result_g atol=atol

                    # ๐ซ - ๐ซ โ ๐
                    @test mul_g(f, g) โ result_g atol=atol
                    @test mul_g(g, f) โ result_g atol=atol
                    @test mul_g(fฬ, g) โ mul_g(f, gฬ) atol=atol

                    @test gฬ * f โ fฬ * g atol=atol
                end

                @testset "f(๐ซ)|fฬ(๐) / g(๐ซ)|gฬ(๐)" begin
                    shell_result = shell + shell_other > max_shells ? InfOrder : shell + shell_other
                    result_r1 = ScalarFieldR{T, TR, TG}(f.basis, shell_result, f.r_data .* inv.(g.r_data)) # f / g
                    result_r2 = ScalarFieldR{T, TR, TG}(f.basis, shell_result, g.r_data .* inv.(f.r_data)) # g / f
                    result_g1 = ๐(result_r1)
                    result_g2 = ๐(result_r2)
    
                    # ๐ซ + ๐ซ โ ๐ซ
                    @test div_r(f, g) โ result_r1 atol=atol
                    @test div_r(g, f) โ result_r2 atol=atol
                    # ๐ซ + ๐ โ ๐ซ
                    @test div_r(f, gฬ) โ result_r1 atol=atol
                    @test div_r(gฬ, f) โ result_r2 atol=atol
                    # ๐ + ๐ซ โ ๐ซ
                    @test div_r(fฬ, g) โ result_r1 atol=atol
                    @test div_r(g, fฬ) โ result_r2 atol=atol
                    # ๐ + ๐  โ ๐ซ
                    @test div_r(fฬ, gฬ) โ result_r1 atol=atol
                    @test div_r(gฬ, fฬ) โ result_r2 atol=atol
    
                    @test div_r(fฬ, g) โ div_r(f, gฬ) atol=atol
                    @test div_r(g, fฬ) โ div_r(gฬ, f) atol=atol
                    @test gฬ /แตฃ f โ g /แตฃ fฬ atol=atol

                    # ๐ - ๐ โ ๐
                    @test div_g(fฬ, gฬ) โ result_g1 atol=atol
                    @test div_g(gฬ, fฬ) โ result_g2 atol=atol

                    # ๐ซ - ๐ โ ๐
                    @test div_g(f, gฬ) โ result_g1 atol=atol
                    @test div_g(gฬ, f) โ result_g2 atol=atol

                    # ๐ - ๐ซ โ ๐
                    @test div_g(fฬ, g) โ result_g1 atol=atol
                    @test div_g(g, fฬ) โ result_g2 atol=atol

                    # ๐ซ - ๐ซ โ ๐
                    @test div_g(f, g) โ result_g1 atol=atol
                    @test div_g(g, f) โ result_g2 atol=atol
                    
                    @test gฬ / f โ g / fฬ atol=atol
                    @test g /แตฃ f โ inv(fฬ /แตฃ gฬ) atol=atol 
                end
            end
        end
    end
end