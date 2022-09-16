

using BenchmarkTools

module EEDFT
    

include("types.jl")
include("atomistics/species.jl")
include("atomistics/atoms.jl")
include("basis.jl")
include("scalar_field.jl")
include("fft.jl")
include("bivector.jl")


export species
export Atoms
export ScalarField, ScalarFieldR, ScalarFieldG

export PlaneWaveBasis
export r_vectors, r_vectors_cart
export G_vectors, G_vectors_cart

export integrate
export fft_plan_data_types


𝔉 = r_to_G
𝔉! = r_to_G!

𝔉⁻¹ = G_to_r
𝔉⁻¹! = G_to_r!

export 𝔉, 𝔉!, 𝔉⁻¹, 𝔉⁻¹!, r_to_G, r_to_G!, G_to_r, G_to_r!
export extract, fft_plan_data_types, unpack!, sizeof, InfOrder, inforder
export add_g, add_r, (+ᵣ)
export sub_g, sub_r, (-ᵣ)
export mul_g, mul_r, (*ᵣ)
export div_g, div_r, (/ᵣ)
export inv_g, inv_r, inv
export dot_g, dot_r, (⋅), (⋅ᵣ)
export isapprox, (≈)
export diff_g, diff_r, (∂ᵢ), (∂ⱼ), (∂ₖ),(∂ᵢʳ), (∂ⱼʳ), (∂ₖʳ), (∇),  (∇ᵣ), (Δ), (Δᵣ)
end