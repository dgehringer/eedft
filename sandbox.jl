using FFTW
using EEDFT
using BenchmarkTools

# FFTW.set_num_threads(6)

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


function compute_χ_g(ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Pᵢ, Pⱼ, Pₖ)
    Sⱼ¹²Sₖ¹² = Sⱼ¹² *Sₖ¹²
    Sₖ¹²Sᵢ¹² = Sₖ¹² * Sᵢ¹²
    Sᵢ¹²Sⱼ¹² = Sᵢ¹² * Sⱼ¹²

    ρ¹²Sᵢ¹² = ρ¹² * Sᵢ¹²
    ρ¹²Sⱼ¹² = ρ¹² * Sⱼ¹²
    ρ¹²Sₖ¹² = ρ¹² * Sₖ¹²
    
    χᵢ⁺ = (Sⱼ¹²Sₖ¹² + ρ¹²Sᵢ¹²) * Pᵢ
    χᵢ⁻ = (Sⱼ¹²Sₖ¹² - ρ¹²Sᵢ¹²) * Pᵢ

    χⱼ⁺ = (Sₖ¹²Sᵢ¹² + ρ¹²Sⱼ¹²) * Pⱼ
    χⱼ⁻ = (Sₖ¹²Sᵢ¹² - ρ¹²Sⱼ¹²) * Pⱼ

    χₖ⁺ = (Sᵢ¹²Sⱼ¹² + ρ¹²Sₖ¹²) * Pₖ
    χₖ⁻ = (Sᵢ¹²Sⱼ¹² - ρ¹²Sₖ¹²) * Pₖ

    return (χᵢ⁺, χᵢ⁻, χⱼ⁺, χⱼ⁻, χₖ⁺, χₖ⁻)
end 

function compute_bivector_g(ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹²)
    ρ, Sᵢ, Sⱼ, Sₖ = map((d¹²) -> d¹² * d¹², (ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹²))
    S = Sᵢ + Sⱼ + Sₖ

    Pᵢ, Pⱼ, Pₖ = map((Sₖ′) -> inv_g((ρ-S)/2.0 + Sₖ), (Sᵢ, Sⱼ, Sₖ))
    χᵢ⁺, χᵢ⁻, χⱼ⁺, χⱼ⁻, χₖ⁺, χₖ⁻ = compute_χ_g(ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Pᵢ, Pⱼ, Pₖ)
    return χᵢ⁺
end

function compute_χ_r(ρ¹², 𝐒¹², 𝐏)
    Pᵢ, Pⱼ, Pₖ = 𝐏
    Sᵢ¹², Sⱼ¹², Sₖ¹² = 𝐒¹²

    Sⱼ¹²Sₖ¹² = Sⱼ¹² *ᵣ Sₖ¹²
    Sₖ¹²Sᵢ¹² = Sₖ¹² *ᵣ Sᵢ¹²
    Sᵢ¹²Sⱼ¹² = Sᵢ¹² *ᵣ Sⱼ¹²

    ρ¹²Sᵢ¹² = ρ¹² *ᵣ Sᵢ¹²
    ρ¹²Sⱼ¹² = ρ¹² *ᵣ Sⱼ¹²
    ρ¹²Sₖ¹² = ρ¹² *ᵣ Sₖ¹²

    χᵢ⁺ = (Sⱼ¹²Sₖ¹² +ᵣ ρ¹²Sᵢ¹²) *ᵣ Pᵢ
    χᵢ⁻ = (Sⱼ¹²Sₖ¹² -ᵣ ρ¹²Sᵢ¹²) *ᵣ Pᵢ

    χⱼ⁺ = (Sₖ¹²Sᵢ¹² +ᵣ ρ¹²Sⱼ¹²) *ᵣ Pⱼ
    χⱼ⁻ = (Sₖ¹²Sᵢ¹² -ᵣ ρ¹²Sⱼ¹²) *ᵣ Pⱼ

    χₖ⁺ = (Sᵢ¹²Sⱼ¹² +ᵣ ρ¹²Sₖ¹²) *ᵣ Pₖ
    χₖ⁻ = (Sᵢ¹²Sⱼ¹² -ᵣ ρ¹²Sₖ¹²) *ᵣ Pₖ

    return (χᵢ⁺, χᵢ⁻, χⱼ⁺, χⱼ⁻, χₖ⁺, χₖ⁻)
end

function compute_B_r(𝛘)
    χᵢ⁺, χᵢ⁻, χⱼ⁺, χⱼ⁻, χₖ⁺, χₖ⁻ = 𝛘

    𝛘′ = ((χᵢ⁺, χᵢ⁻), (χⱼ⁺, χⱼ⁻), (χₖ⁺, χₖ⁻))

    Dᵢ, Dⱼ, Dₖ = ( inv_r(1.0 -ᵣ (χ⁻ *ᵣ χ⁺))  for (χ⁺, χ⁻) in 𝛘′)

    Bᵢᵏ = (χₖ⁻ -ᵣ (χⱼ⁺ *ᵣ χᵢ⁺)) *ᵣ Dᵢ
    Bᵢʲ = (χⱼ⁺ -ᵣ (χₖ⁻ *ᵣ χᵢ⁻)) *ᵣ Dᵢ

    Bⱼⁱ = (χᵢ⁻ -ᵣ (χₖ⁺ *ᵣ χⱼ⁺)) *ᵣ Dⱼ
    Bⱼᵏ = (χₖ⁺ -ᵣ (χᵢ⁻ *ᵣ χⱼ⁻)) *ᵣ Dⱼ

    Bₖʲ = (χⱼ⁻ -ᵣ (χᵢ⁺ *ᵣ χₖ⁺)) *ᵣ Dₖ
    Bₖⁱ = (χᵢ⁺ -ᵣ (χⱼ⁻ *ᵣ χₖ⁻)) *ᵣ Dₖ

    return (Bᵢᵏ, Bᵢʲ, Bⱼⁱ, Bⱼᵏ, Bₖʲ, Bₖⁱ)
end


function compute_𝐉̂_r(ρ¹², 𝐒¹², 𝐏)
    ∇Sᵢ¹², ∇Sⱼ¹², ∇Sₖ¹² = map(∇ᵣ, 𝐒¹²)
end

function compute_bivector_r(ρ¹², 𝐒¹²)
    Sᵢ¹², Sⱼ¹², Sₖ¹² = 𝐒¹²
    ρ, Sᵢ, Sⱼ, Sₖ = map((d¹²) -> d¹² * d¹², (ρ¹², 𝐒¹²...))
    𝐒 = Sᵢ, Sⱼ, Sₖ
    S = Sᵢ + Sⱼ + Sₖ

    𝐏 = map((Sₖ′) -> inv_r((ρ-S)/2.0 + Sₖ), (Sᵢ, Sⱼ, Sₖ))
    𝛘 = compute_χ_r(ρ¹², 𝐒¹², 𝐏)

    𝐁 = compute_B_r(𝛘)



    𝐉̂ = compute_𝐉̂_r(ρ¹², 𝐒¹², 𝐏)

    return 1
end


1