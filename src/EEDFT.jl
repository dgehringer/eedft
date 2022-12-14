

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


π = r_to_G
π! = r_to_G!

πβ»ΒΉ = G_to_r
πβ»ΒΉ! = G_to_r!

export π, π!, πβ»ΒΉ, πβ»ΒΉ!, r_to_G, r_to_G!, G_to_r, G_to_r!
export extract, fft_plan_data_types, unpack!, sizeof, InfOrder, inforder
export add_g, add_r, (+α΅£)
export sub_g, sub_r, (-α΅£)
export mul_g, mul_r, (*α΅£)
export div_g, div_r, (/α΅£)
export inv_g, inv_r, inv
export dot_g, dot_r, (β), (βα΅£)
export isapprox, (β)
export diff_g, diff_r, (βα΅’), (ββ±Ό), (ββ),(βα΅’Κ³), (ββ±ΌΚ³), (ββΚ³), (β),  (βα΅£), (Ξ), (Ξα΅£)
end