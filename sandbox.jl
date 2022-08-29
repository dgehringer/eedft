
# include("src/EEDFT.jl")

using EEDFT
using FFTW

fcc_al = Atoms([1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0], [0.0 0.0 0.0; 0.5 0.5 0.0; 0.0 0.5 0.5; 0.5 0.0 0.5], species(:Ni, :Al, :Ni, :Ni))

# basis_c = PlaneWaveBasis(fcc_al, 100.0, ComplexF64)
basis_f = PlaneWaveBasis(fcc_al, 100.0, Float64)

function make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int, fft_func::Function) where {T, TR}
    _, TG = fft_plan_data_types(basis.fft_plan_fw)
    r_data = rand(Tr, basis.r_grid_size) * Tr(100.0)
    ggrid = fft_func(r_data)
    g_data = extract(ScalarField{T, TR, TG}(basis, order, nothing, r_data), ggrid)
    ScalarField{T, TR, TG}(basis, order, g_data, nothing)
end

make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int)  where {T, TR <: Real} = make_scalar_field(Tr, basis, order, rfft)
make_scalar_field(Tr::Type{TR}, basis::PlaneWaveBasis{T, TR}, order::Int)  where {T, TR <: Complex} = make_scalar_field(Tr, basis, order, fft)


f = make_scalar_field(Float64, basis_f, 1)

function integrate_rfft_G(spectrum::Array{T, N}, n::Union{Int, Nothing}=nothing) where {T <: Complex, N}
    """
    Use Parseval's theorem to find the RMS value of an even-length signal
    from its rfft, without wasting time doing an inverse real FFT.
    spectrum is produced as spectrum = numpy.fft.rfft(signal)
    For a signal x with an even number of samples, these should produce the
    same result, to within numerical accuracy:
    rms_flat(x) ~= rms_rfft(rfft(x))
    If len(x) is odd, n must be included, or the result will only be
    approximate, due to the ambiguity of rfft for odd lengths.
    """

    rdims = collect((:) for i in 1:ndims(spectrum)-1)

    if isnothing(n)
        n = (first(size(spectrum)) - 1) * 2
    end
    sq = real(spectrum .* conj(spectrum))
    fft_size = prod(tuple(n, size(spectrum)[2:end]...))
    if n % 2 == 0  # even-length
        mean = (sum(sq[1, rdims...]) + 2.0*sum(sq[2:end-1, rdims...]) + sum(sq[end, rdims...])) / fft_size
    else  # odd-length
        mean = (sum(sq[1, rdims...]) + 2.0*sum(sq[2:end, rdims...])) / fft_size
    end

    return mean
end

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
