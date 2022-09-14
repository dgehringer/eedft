

using IterTools
import Base.show

@doc raw"""
Here is some documentation
"""
struct PlaneWaveBasis{T <: Real, V}

    r_grid_size::Tuple{Integer, Integer, Integer}
    G_grid_size::Tuple{Integer, Integer, Integer}
    Ecut::T
    atoms::Atoms

    dΩ::T
    r_to_G_norm::T
    G_to_r_norm::T

    fft_plan_fw
    fft_plan_bw

    G_shell_num_waves::Dict{Int, Int}
    G_shell_indices::Dict{Int, Vector{CartesianIndex}}
    G_shell_indices_flat::Dict{Int, Vector{Int}}
    G_shell_vectors::Dict{Int, Matrix{T}}
    G_shell_norms::Dict{Int, Vector{T}}
    G_shell_upscale_maps::Dict{Pair{Int, Int}, Vector{Int}}
end


function compute_G_shells(Gshells, Ecut::T, reciprocal_cell::Mat3{T}, fft_grid_size::Tuple{Int, Int, Int}, data_type::Type{V})  where {T, V}
    shell_usage = Dict{Int, Int}(shell => 0 for shell in Gshells)

    Gcuts = [(shell, shell*sqrt(2Ecut)) for shell in reverse(Gshells)]
    shell_indices = Dict{Int, Vector{CartesianIndex}}(shell => [] for shell in Gshells)
    flat_indices = Dict{Int, Vector{Int}}(shell => [] for shell in Gshells)
    shell_vectors = Dict{Int, Vector{Vec3{T}}}(shell => [] for shell in Gshells)
    shell_norms = Dict{Int, Vector{T}}(shell => [] for shell in Gshells)
    shell_upscale_map = Dict{Pair{Int, Int}, Vector{Int}}((shell_low => shell_high) => [] for shell_low in Gshells for shell_high in (shell_low+1):max(Gshells...))
    shell_upscale_ptrs = Dict{Int, Int}(shell => 1 for shell in Gshells)
 
    for (idx, (Gi, G)) in enumerate(G_vectors_with_indices(data_type, fft_grid_size))
        for (shell, Gmax) in Gcuts
            Gcart = reciprocal_cell * G
            if norm(Gcart) ≤ Gmax 
                shell_usage[shell] += 1
                push!(shell_indices[shell], Gi) 
                push!(flat_indices[shell], idx)
                push!(shell_vectors[shell], Gcart)
                push!(shell_norms[shell], norm(Gcart))
                shell_upscale_ptrs[shell] = length(flat_indices[shell])
                for other_shell in reverse((shell+1):maximum(Gshells))
                    push!(shell_upscale_map[shell => other_shell], shell_upscale_ptrs[other_shell])
                end
            end
        end
    end

    shell_vectors = Dict{Int, Matrix{T}}(shell => hcat(Vector.(Gs)...) for (shell, Gs) in shell_vectors)
    # sanity checks that the generated indices in the upscale maps are okay
    for ((lower_shell, upper_shell), indices) in shell_upscale_map
        @assert length(indices) == length(flat_indices[lower_shell])
        @assert maximum(indices) <= length(flat_indices[upper_shell])
        @assert all(flat_indices[lower_shell] .== flat_indices[upper_shell][indices])
    end

    shell_indices, flat_indices, shell_usage, shell_vectors, shell_norms, shell_upscale_map
end


function PlaneWaveBasis(atoms::Atoms{T}, Ecut::T, K::Type{V}; fft_grid_size=nothing, Gshells=1:4, fftw_flags=FFTW.MEASURE) where {T, V}
    
    if isnothing(fft_grid_size)
        fft_grid_size = compute_fft_size(atoms.cell, Ecut; supersampling=maximum(Gshells))
    end

    out_fft_plan, out_back_fft_plan, r_grid_size, G_grid_size = build_fft_plans(K, fft_grid_size; flags=fftw_flags)
    
    basis_set_size = prod(fft_grid_size)
    (shell_indices, flat_indices, shell_usage, shell_vectors, shell_norms, shell_upscale_map) = compute_G_shells(Gshells, Ecut, atoms.reciprocal_cell, fft_grid_size, K)

    dΩ = atoms.volume/basis_set_size
    G_to_r_norm = 1/sqrt(atoms.volume)
    r_to_G_norm = sqrt(atoms.volume) / basis_set_size


    PlaneWaveBasis{T, V}(r_grid_size, G_grid_size, Ecut, atoms, dΩ, r_to_G_norm, G_to_r_norm, out_fft_plan, 
        out_back_fft_plan, shell_usage, shell_indices, flat_indices, shell_vectors, shell_norms, shell_upscale_map)
end


Base.show(io::IO, x::PlaneWaveBasis{T}) where T = print(io, "PlaneWaveBasis(Ecut=$(x.Ecut), G_shells=$(x.G_shell_num_waves))") 


# enhance utility functions defined in "src/fft.jl"

G_vectors(basis::PlaneWaveBasis{T, V}) where {T, V} = G_vectors(V, basis.r_grid_size)

G_vectors_with_indices(basis::PlaneWaveBasis{T, V}) where {T, V} = G_vectors_with_indices(V, basis.r_grid_size)

G_vectors_cart(basis::PlaneWaveBasis) = (basis.atoms.reciprocal_cell * G for G in G_vectors(basis))

r_vectors(basis::PlaneWaveBasis{T, V}) where {T, V} = r_vectors(T, basis.r_grid_size)

r_vectors_cart(basis::PlaneWaveBasis{T, V}) where {T, V} = (basis.atoms.cell * r for r in r_vectors(basis))

# utility functions to extract parametric types from PlaneWaveBasis{T, V}

basis_grid_data_type(basis::PlaneWaveBasis{T, V}) where {T, V} = V

basis_internal_data_type(basis::PlaneWaveBasis{T, V}) where {T, V} = T

function G_to_r!(f_fourier::Array{TG, 3}, basis::PlaneWaveBasis{T, TR}, f_real::Array{TR, 3}; normalize::Bool=true) where {T, TR, TG}
    G_to_r!(f_fourier, basis.fft_plan_bw, f_real)
    if normalize
        f_real[:] = f_real .* basis.G_to_r_norm
    end
    f_real
end

function G_to_r(f_fourier::Array{TG, 3}, basis::PlaneWaveBasis{T, TR}; normalize::Bool=true) where {T, TR, TG}
    f_real = G_to_r(f_fourier, basis.fft_plan_bw)
    normalize ? f_real .* basis.G_to_r_norm : f_real
end

function r_to_G!(f_real::Array{TR, 3}, basis::PlaneWaveBasis{T, TR}, f_fourier::Array{TG, 3}; normalize::Bool=true) where {T, TR, TG}
    r_to_G!(f_real, basis.fft_plan_fw, f_fourier)
    if normalize
        f_fourier[:] = f_fourier .* basis.r_to_G_norm
    end
    f_fourier
end

function r_to_G(f_real::Array{TR, 3}, basis::PlaneWaveBasis{T, TR}; normalize::Bool=true) where {T, TR}
    f_fourier = r_to_G(f_real, basis.fft_plan_fw)
    normalize ? f_fourier .* basis.r_to_G_norm : f_fourier
end
