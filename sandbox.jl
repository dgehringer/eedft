using Pkg

Pkg.activate(".")

using EEDFT
using FFTW

fcc_al = Atoms([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0], [0.0 0.0 0.0; 0.5 0.5 0.0; 0.0 0.5 0.5; 0.5 0.0 0.5], species(:Ni, :Al, :Ni, :Ni))

# basis_c = PlaneWaveBasis(fcc_al, 100.0, ComplexF64)
basis_f = PlaneWaveBasis(fcc_al, 1000.0, Float64)


function unpack_rfft_grid!(a::Array{T, N}, to::Array{T, N}) where {T <: Complex, N}
    
    rdims = collect((:) for i in 1:ndims(a)-1)
    n = first(size(to))
    fill!(to, T(0.0))

    nr = first(size(a))
    to[1:nr, rdims...] = a
    if n % 2 == 0  # even-length
        to[end:-1:(nr+1), rdims...] = a[2:end-1, rdims...]
    else
        to[end:-1:(nr+1), rdims...] = a[2:end, rdims...]
    end
    to
end

function unpack_rfft_grid(a::Array{T, N}, n::Union{Int, Nothing}=nothing) where {T <: Complex, N}
    if isnothing(n)
        n = (first(size(a)) - 1) * 2
    end
    fft_size = tuple(n, size(a)[2:end]...)
    to = Array{T, 3}(undef, fft_size...)
    unpack_rfft_grid!(a, to)
    to
end

shell = 1
shellp = 1

Gv = transpose(basis_f.G_shell_vectors[shell])
Gvp = transpose(basis_f.G_shell_vectors[shellp])
Gvr = transpose(basis_f.G_shell_vectors[shell + shellp])

mult = Dict{Vector{Float64}, Vector{Tuple{Int, Int}}}()

