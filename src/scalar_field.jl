
import Base: -, +,*,inv,/
    

mutable struct ScalarField{T <: Real, TR, TG}
    basis::PlaneWaveBasis{T, TR}
    order::Int
    g_data::Union{Vector{TG}, Nothing}
    r_data::Union{Array{TR, 3}, Nothing}
end



ScalarField(basis::PlaneWaveBasis{T, TR}, order::Int, g_data::Vector{TG}) where {T, TR, TG} = ScalarField{T, TR, TG}(basis, order, g_data, nothing)

function ScalarField(basis::PlaneWaveBasis{T, TR}, order::Int, r_data::Array{TR, 3}) where {T, TR}
    TR_, TG = fft_plan_data_types(basis.fft_plan_fw)
    @assert TR_ == TR
    ScalarField{T, TR, TG}(basis, order, nothing, r_data)
end


function extract(f::ScalarField{T, TR, TG}, f_fourier::Array{TG, 3}) where {T, TR, TG}
    flattened = reshape(f_fourier, prod(f.basis.G_grid_size))
    @assert size(f_fourier) == f.basis.G_grid_size
    flattened[f.basis.G_shell_indices_flat[f.order]]
end


function unpack!(f::ScalarField{T, TR, TG}, f_fourier::Array{TG, 3}) where {T, TR, TG}
    fill!(f_fourier, TG(0.0))
    @assert size(f_fourier) == f.basis.G_grid_size
    flattened = reshape(f_fourier, prod(f.basis.G_grid_size))
    flattened[f.basis.G_shell_indices_flat[f.order]] = f.g_data
    reshape(flattened, f.basis.G_grid_size)
end

ensure_G_space(f::ScalarField{T, TR, TG}) where {T, TR, TG} = isnothing(f.g_data) && throw(DomainError("ScalarField is already represented in real space"))
ensure_r_space(f::ScalarField{T, TR, TG}) where {T, TR, TG} = isnothing(f.r_data) && throw(DomainError("ScalarField is already represented in reciprocal space"))


function G_to_r!(f::ScalarField{T, TR, TG}, f_fourier::Array{TG, 3}, f_real::Array{TR, 3}; normalize::Bool=true) where {T, TG, TR}
    ensure_G_space(f)
    f_fourier = unpack!(f, f_fourier)
    G_to_r!(f_fourier, f.basis, f_real; normalize=normalize)
    f.r_data = f_real
    f.g_data = nothing
    f
end

function G_to_r!(f::ScalarField{T, TR, TG}, f_real::Array{TR, 3}; normalize::Bool=true) where {T, TR, TG}
    f_fourier = Array{TG, 3}(undef, f.basis.G_grid_size)
    G_to_r!(f, f_fourier, f_real; normalize=normalize)
end

function G_to_r!(f::ScalarField{T, TR, TG}; normalize::Bool=true) where {T, TR, TG}
    f_real = Array{TR, 3}(undef, f.basis.r_grid_size)
    G_to_r!(f, f_real; normalize=normalize)
end

function G_to_r(f::ScalarField{T, TR, TG}; normalize::Bool=true) where {T, TR, TG}
    ensure_G_space(f)
    f_fourier = Array{TG, 3}(undef, f.basis.G_grid_size)
    unpack!(f, f_fourier)
    ScalarField{T, TR, TG}(f.basis, f.order, nothing, G_to_r(f_fourier, f.basis; normalize=normalize))
end

function r_to_G!(f::ScalarField{T, TR, TG}, f_fourier::Array{TG, 3}; normalize::Bool=true) where {T, TG, TR}
    ensure_r_space(f)
    r_to_G!(f.r_data, f.basis, f_fourier; normalize=normalize)
    f.g_data = extract(f, f_fourier)
    f.r_data = nothing
    f
end

function r_to_G!(f::ScalarField{T, TR, TG}; normalize::Bool=true) where {T, TG, TR}
    f_fourier = Array{TG, 3}(undef, f.basis.G_grid_size)
    r_to_G!(f, f_fourier; normalize=normalize)
end

function r_to_G(f::ScalarField{T, TR, TG}; normalize::Bool=true) where {T, TG, TR}
    ensure_r_space(f)
    f_fourier = r_to_G(f.r_data, f.basis; normalize=normalize)
    ScalarField(f.basis, f.order, extract(f, f_fourier), nothing)
end


is_r(f::ScalarField) = !isnothing(f.r_data)
is_G(f::ScalarField) = !isnothing(f.g_data)

function integrate_rfft_G(spectrum::Array{TG, 3}, n::Union{Nothing, Int}=nothing) where {TG  <: Complex}
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
    if n % 2 == 0  # even-length
        mean = (sum(sq[1, rdims...]) + 2.0*sum(sq[2:end-1, rdims...]) + sum(sq[end, rdims...]))
    else  # odd-length
        mean = (sum(sq[1, rdims...]) + 2.0*sum(sq[2:end, rdims...]))
    end

    return mean
end

function integrate_rfft_G(f::ScalarField{T, TR, TG}) where {T, TR, TG  <: Complex}
    g_grid = Array{TG, 3}(undef, f.basis.G_grid_size)
    unpack!(f, g_grid)
    integrate_rfft_G(g_grid, first(f.basis.r_grid_size))
end


sum_G_square(a::Array{T}) where {T <: Complex} = sum(real(a .* conj(a))) 

sum_G_(a::Array{T}) where {T <: Complex} = sqrt(sum_G_square(a))

integrate(f::ScalarField{T, TR, TG}) where {T, TR <: Real, TG} = isnothing(f.g_data) ? sum(f.r_data .^ 2) / prod(f.basis.r_grid_size) : integrate_rfft_G(f)
integrate(f::ScalarField{T, TR, TG}) where {T, TR <: Complex, TG} = isnothing(f.g_data) ? sum(real(f.r_data .* conj(f.r_data))) / prod(f.basis.r_grid_size) : sum(real(f.g_data .* conj(f.g_data)))

function add!(f::ScalarField{T, TR, TG}, g::ScalarField{T, TR, TG}) where {T, TR, TG}
end

