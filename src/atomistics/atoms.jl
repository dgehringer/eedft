
using IterTools
using LinearAlgebra

field(f::Symbol) = (o::Any) -> getfield(o, f)

concat(fs...) = (o::Any) -> [f(o) for f in fs]

inbounds(a::Number, l::Number, u::Number) = (a ≈ l || a > l) && (a ≈ u || a < u)

in_unitcell(a::AbstractFloat) = 0.0 <= a < 1.0
in_unitcell(a::Vec3{T}) where {T <: AbstractFloat} = all(map(in_unitcell, a))

function to_unitcell(a::AbstractFloat)
   a %= 1.0
   a ≈ 1.0 && return a - 1.0
   a < 0.0 && return 1.0 + a
   a
end

to_unitcell(a::Vec3{T}) where {T <: AbstractFloat} = map(to_unitcell, a)

struct Atoms{T <: Real}
   cell::Mat3{T}
   scaled_positions::Vector{Vec3{T}}
   species::Vector{Species}

   volume::T
   reciprocal_volume::T
   reciprocal_cell::Mat3{T}
   positions::Vector{Vec3{T}}
end

function Atoms(cell::Mat3{T}, scaled_positions::Vector{Vec3{T}}, atoms::Vector{Species}) where {T <: Real}
   _is_well_conditioned(cell) || @warn("Your lattice is conditions badly")
   volume = abs(det(cell))
   
   for (i, pos) in enumerate(scaled_positions)
      need_wrapping = pos .|> in_unitcell |> all |> !
      
      if need_wrapping
         wrapped = pos .|> to_unitcell
         @info("wrapping lattice position $i: $pos → $wrapped")
         scaled_positions[i] = wrapped
      end
      
   end
   tcell = transpose(cell)
   positions = collect(tcell * pos for pos in scaled_positions)
   reciprocal_cell = 2T(π) * transpose(inv(cell))
   reciprocal_volume = abs(det(reciprocal_cell))
   Atoms{T}(cell, scaled_positions, atoms, volume, reciprocal_volume, reciprocal_cell, positions)
end


function Atoms(cell::Matrix{T}, scaled_positions::Matrix{T}, atoms::Vector{Species}) where {T <: Real}
   size(cell) == (3, 3) || @error("The lattice must be specified as a 3x3 matrix")
   size(atoms, 1) == size(scaled_positions, 1) || @error("The number of atoms does not match the number of lattice positions")
   size(scaled_positions, 2) == 3 || @error("The position matrix must be of shape N x 3") 
   Atoms(Mat3{T}(cell), map(Vec3{T}, eachrow(scaled_positions)), atoms)
end

function chemical_formula(a::Atoms{T}) where {T}
   grouped_species = collect(groupby(field(:ordinal), a.species))
   histogram = grouped_species .|> concat(String ∘ field(:symbol) ∘ first, length)
   formula = join( num == 1 ? sym : join((sym, num)) for (sym, num) in histogram)
   replace(formula, r"\d" => (digit) -> '₀' + parse(Int, digit))   
end

_is_well_conditioned(A; tol=1e5) = (cond(A) <= tol)
