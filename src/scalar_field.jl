
import Base: -, +,*,inv,/, Callable,(==), isapprox
    

struct ScalarFieldG{T <: Real, TR, TG}
    basis::PlaneWaveBasis{T, TR}
    order::Int
    g_data::Vector{TG}
end

struct ScalarFieldR{T <: Real, TR, TG}
    basis::PlaneWaveBasis{T, TR}
    order::Int
    r_data::Array{TR, 3}
end

const ScalarField{T, TR, TG} = Union{ScalarFieldR{T, TR, TG}, ScalarFieldG{T, TR, TG}} where {T, TR, TG}

ScalarField(basis::PlaneWaveBasis{T, TR}, order::Int, g_data::Vector{TG}) where {T, TR, TG} = ScalarFieldG{T, TR, TG}(basis, order, g_data)

function ScalarField(basis::PlaneWaveBasis{T, TR}, order::Int, r_data::Array{TR, 3}) where {T, TR}
    TR_, TG = fft_plan_data_types(basis.fft_plan_fw)
    @assert TR_ == TR
    ScalarFieldR{T, TR, TG}(basis, order, r_data)
end

function ==(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG}
    f.order != g.order && return false
    f.basis != g.base && return false
    f.r_data == f.r_data
end

function ==(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG}
    f.order != g.order && return false
    f.basis != g.basis && return false
    f.g_data == f.g_data
end

==(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = f == 𝔉(g)
==(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = 𝔉(f) == g

function isapprox(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldR{T, TR, TG}, args...; kwargs...) where {T, TR, TG}
    f.order != g.order && return false
    f.basis != g.basis && return false
    isapprox(f.r_data, f.r_data, args...; kwargs...)
end

function isapprox(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldG{T, TR, TG}, args...; kwargs...) where {T, TR, TG}
    f.order != g.order && return false
    f.basis != g.basis && return false
    isapprox(f.g_data, f.g_data, args...; kwargs...)
end

isapprox(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldR{T, TR, TG}, args...; kwargs...) where {T, TR, TG} = isapprox(f, 𝔉(g), args...; kwargs...)
isapprox(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldG{T, TR, TG}, args...; kwargs...) where {T, TR, TG} = isapprox(𝔉(f), g, args...; kwargs...)

function extract(f::ScalarField{T, TR, TG}, f_fourier::Array{TG, 3}) where {T, TR, TG}
    flattened = reshape(f_fourier, prod(f.basis.G_grid_size))
    @assert size(f_fourier) == f.basis.G_grid_size
    flattened[f.basis.G_shell_indices_flat[f.order]]
end


function unpack!(f::ScalarFieldG{T, TR, TG}, f_fourier::Array{TG, 3}) where {T, TR, TG}
    fill!(f_fourier, TG(0.0))
    @assert size(f_fourier) == f.basis.G_grid_size
    flattened = reshape(f_fourier, prod(f.basis.G_grid_size))
    flattened[f.basis.G_shell_indices_flat[f.order]] = f.g_data
    reshape(flattened, f.basis.G_grid_size)
end


function G_to_r!(f::ScalarFieldG{T, TR, TG}, f_fourier::Array{TG, 3}, f_real::Array{TR, 3}; normalize::Bool=true) where {T, TG, TR}
    f_fourier = unpack!(f, f_fourier)
    G_to_r!(f_fourier, f.basis, f_real; normalize=normalize)
    ScalarFieldR{T, TR, TG}(f.basis, f.order, f_real)
end

function G_to_r!(f::ScalarFieldG{T, TR, TG}, f_real::Array{TR, 3}; normalize::Bool=true) where {T, TR, TG}
    f_fourier = Array{TG, 3}(undef, f.basis.G_grid_size)
    G_to_r!(f, f_fourier, f_real; normalize=normalize)
end

function G_to_r!(f::ScalarFieldG{T, TR, TG}; normalize::Bool=true) where {T, TR, TG}
    f_real = Array{TR, 3}(undef, f.basis.r_grid_size)
    G_to_r!(f, f_real; normalize=normalize)
end

function G_to_r(f::ScalarFieldG{T, TR, TG}; normalize::Bool=true) where {T, TR, TG}
    f_fourier = Array{TG, 3}(undef, f.basis.G_grid_size)
    unpack!(f, f_fourier)
    ScalarFieldR{T, TR, TG}(f.basis, f.order, G_to_r(f_fourier, f.basis; normalize=normalize))
end

function r_to_G!(f::ScalarFieldR{T, TR, TG}, f_fourier::Array{TG, 3}; normalize::Bool=true) where {T, TG, TR}
    r_to_G!(f.r_data, f.basis, f_fourier; normalize=normalize)
    ScalarFieldG{T, TR, TG}(f.basis, f.order, extract(f, f_fourier))
end

function r_to_G!(f::ScalarFieldR{T, TR, TG}; normalize::Bool=true) where {T, TG, TR}
    f_fourier = Array{TG, 3}(undef, f.basis.G_grid_size)
    r_to_G!(f, f_fourier; normalize=normalize)
end

function r_to_G(f::ScalarFieldR{T, TR, TG}; normalize::Bool=true) where {T, TG, TR}
    ScalarFieldG{T, TR, TG}(f.basis, f.order, extract(f, r_to_G(f.r_data, f.basis; normalize=normalize)))
end


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

function integrate_rfft_G(f::ScalarFieldG{T, TR, TG}) where {T, TR, TG  <: Complex}
    g_grid = Array{TG, 3}(undef, f.basis.G_grid_size)
    unpack!(f, g_grid)
    integrate_rfft_G(g_grid, first(f.basis.r_grid_size))
end


sum_G_square(f::ScalarFieldR{T, TR, TG}) where {T, TR <: Real, TG} = sum(f.r_data .^ 2) / prod(f.basis.r_grid_size) 
sum_G_square(f::ScalarFieldG{T, TR, TG}) where {T, TR <: Real, TG} = integrate_rfft_G(f)
sum_G_square(f::ScalarFieldR{T, TR, TG}) where {T, TR <: Complex, TG} = sum(real(f.r_data .* conj(f.r_data))) / prod(f.basis.r_grid_size) 
sum_G_square(f::ScalarFieldG{T, TR, TG}) where {T, TR <: Complex, TG} = sum(real(f.g_data .* conj(f.g_data)))


integrate(f::ScalarField{T, TR, TG}) where {T, TR, TG} = sqrt(sum_G_square(f))


# f̃(𝐆) + g̃(𝐆) → h̃(𝐆) 
function add_g(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG}
    f.order == g.order && return ScalarFieldG{T, TR, TG}(f.basis, f.order, f.g_data .+ g.g_data)
    s_up, s_down = ((f.order > g.order) ? (f, g) : (g, f))
    upscale_map = f.basis.G_shell_upscale_maps[s_down.order => s_up.order]
    g_data_up = copy(s_up.g_data)
    g_data_up[upscale_map] = g_data_up[upscale_map] .+ s_down.g_data
    ScalarFieldG{T, TR, TG}(f.basis, max(f.order, g.order), g_data_up)
end

function add_g(a::Number, f::ScalarFieldG{T, TR, TG}) where {T, TR, TG}
    G0_index = findfirst((c) -> c ≈ 0.0, f.basis.G_shell_norms[f.order])
    G0_index === nothing && throw(BoundsError("No constant index"))
    g_data = copy(f.g_data)
    g_data[G0_index] += a
    ScalarFieldG(f.basis, f.order, g_data)
end

add_g(a::Number, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = add_g(a, 𝔉(g))
add_g(g::ScalarFieldG{T, TR, TG}, a::Number) where {T, TR, TG} = add_g(a, g)
add_g(g::ScalarFieldR{T, TR, TG}, a::Number) where {T, TR, TG} = add_g(a, 𝔉(g))

add_g(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = 𝔉(add_r(f, g)) # f(𝐫) + g(𝐫) → h̃(𝐆)
add_g(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = add_g(f, 𝔉(g)) # f̃(𝐆) + g(𝐫) → h̃(𝐆)
add_g(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = add_g(g, f) # f(𝐫) + g̃(𝐆) → h̃(𝐆)

add_r(a::Number, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = ScalarFieldR{T, TR, TG}(g.basis, g.order, g.r_data .+ a)
add_r(a::Number, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = add_r(a, 𝔉⁻¹(g))
add_r(g::ScalarFieldR{T, TR, TG}, a::Number) where {T, TR, TG} = add_r(a, g)
add_r(g::ScalarFieldG{T, TR, TG}, a::Number) where {T, TR, TG} = add_r(a, 𝔉⁻¹(g))

add_r(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = 𝔉⁻¹(add_g(f, g)) # f̃(𝐆) + g̃(𝐆) → h(𝐫)
add_r(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = ScalarFieldR{T, TR, TG}(f.basis, max(f.order, g.order), f.r_data .+ g.r_data) # f(𝐫) + g(𝐫) → h(𝐫)
add_r(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = add_r(𝔉⁻¹(f), g) # f̃(𝐆) + g(𝐫) → h(𝐫)
add_r(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = add_r(g, f) # f(𝐫) + g̃(𝐆) → h(𝐫)

+(f::ScalarField{T, TR, TG}, g::ScalarField{T, TR, TG}) where {T, TR, TG} = add_g(f, g)
+(a::Number, g::ScalarField{T, TR, TG}) where {T, TR, TG} = add_g(a, g)
+(g::ScalarField{T, TR, TG}, a::Number) where {T, TR, TG} = add_g(a, g)
(+ᵣ) = add_r


-(f::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = ScalarFieldR{T, TR, TG}(f.basis, f.order, -f.r_data)
-(f::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = ScalarFieldG{T, TR, TG}(f.basis, f.order, -f.g_data)

sub_g(a::Number, g::ScalarField{T, TR, TG}) where {T, TR, TG} = add_g(a, -g)
sub_g(g::ScalarField{T, TR, TG}, a::Number) where {T, TR, TG} = add_g(g, -a)

sub_g(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = add_g(f, -g) # f̃(𝐆) - g̃(𝐆) → h̃(𝐆) 
sub_g(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = 𝔉(sub_r(f, g)) # f(𝐫) + g(𝐫) → h̃(𝐆)
sub_g(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = sub_g(f, 𝔉(g)) # f̃(𝐆) + g(𝐫) → h̃(𝐆)
sub_g(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = sub_g(𝔉(f), g) # f(𝐫) + g̃(𝐆) → h̃(𝐆)

sub_r(a::Number, g::ScalarField{T, TR, TG}) where {T, TR, TG} = add_r(a, -g)
sub_r(g::ScalarField{T, TR, TG}, a::Number) where {T, TR, TG} = add_r(g, -a)

sub_r(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = 𝔉⁻¹(sub_g(f, g)) # f̃(𝐆) + g̃(𝐆) → h(𝐫)
sub_r(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = ScalarFieldR{T, TR, TG}(f.basis, max(f.order, g.order), f.r_data .- g.r_data) # f(𝐫) + g(𝐫) → h(𝐫)
sub_r(f::ScalarFieldG{T, TR, TG}, g::ScalarFieldR{T, TR, TG}) where {T, TR, TG} = sub_r(𝔉⁻¹(f), g) # f̃(𝐆) + g(𝐫) → h(𝐫)
sub_r(f::ScalarFieldR{T, TR, TG}, g::ScalarFieldG{T, TR, TG}) where {T, TR, TG} = sub_r(f, 𝔉⁻¹(g)) # f(𝐫) + g̃(𝐆) → h(𝐫)

-(f::ScalarField{T, TR, TG}, g::ScalarField{T, TR, TG}) where {T, TR, TG} = sub_g(f, g)
-(a::Number, g::ScalarField{T, TR, TG}) where {T, TR, TG} = sub_g(a, g)
-(g::ScalarField{T, TR, TG}, a::Number) where {T, TR, TG} = sub_g(a, g)
(-ᵣ) = sub_r

