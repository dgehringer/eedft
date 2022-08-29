

function compute_bivector_potential(ρ¹²::ScalarField{T, V}, S¹²::Vec3{ScalarField{T, V}}) where {T, V}
    Sᵢ¹², Sⱼ¹², Sₖ¹² = S¹²
    ρ, Sᵢ, Sⱼ, Sₖ = map(density, (ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹²))
    S = Sᵢ + Sⱼ + Sₖ
    𝐉ᵢ() = 𝐉(ρ, S, ρ¹², Sⱼ¹², Sₖ¹², Sᵢ¹², Sᵢ)
    𝐉ⱼ() = 𝐉(ρ, S, ρ¹², Sₖ¹², Sᵢ¹², Sⱼ¹², Sⱼ)
    𝐉ₖ() = 𝐉(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ)
     
    χᵢ⁺() = χ(ρ, S, ρ¹², Sⱼ¹², Sₖ¹², Sᵢ¹², Sᵢ,+)
    χᵢ⁻() = χ(ρ, S, ρ¹², Sⱼ¹², Sₖ¹², Sᵢ¹², Sᵢ,-)
    χⱼ⁺() = χ(ρ, S, ρ¹², Sₖ¹², Sᵢ¹², Sⱼ¹², Sⱼ,+)
    χⱼ⁻() = χ(ρ, S, ρ¹², Sₖ¹², Sᵢ¹², Sⱼ¹², Sⱼ,-)
    χₖ⁺() = χ(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ,+)
    χₖ⁻() = χ(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ,-)
   
    return χᵢ⁺()
    Bᵢʲᵏ() = B(χᵢ⁺, χᵢ⁻, χⱼ⁺, χₖ⁻)
    Bⱼᵏⁱ() = B(χⱼ⁺, χⱼ⁻, χₖ⁺, χᵢ⁻)
    Bₖⁱʲ() = B(χₖ⁺, χₖ⁻, χᵢ⁺, χⱼ⁻)

    return Bᵢʲᵏ()

    vᵢ() = v(𝐉ⱼ, 𝐉ₖ, 𝐉ᵢ, Bᵢʲᵏ, χⱼ⁺, χₖ⁻)
    vⱼ() = v(𝐉ₖ, 𝐉ᵢ, 𝐉ⱼ, Bⱼᵏⁱ, χₖ⁺, χᵢ⁻)
    vₖ() = v(𝐉ᵢ, 𝐉ⱼ, 𝐉ₖ, Bₖⁱʲ, χᵢ⁺, χⱼ⁻)

    Vec3(vᵢ(), vⱼ(), vₖ())
end

function 𝐉(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ)
   inv(0.5*(ρ-S) + Sₖ) * (ρ¹²*∇²(Sₖ¹²) - Sₖ¹²*∇²(ρ¹²) + Sᵢ¹²*∇²(Sⱼ¹²) + ∇(Sᵢ¹²)⋅∇(Sⱼ¹²) - Sⱼ¹²*∇²(Sⱼ¹²) - ∇(Sⱼ¹²)⋅∇(Sⱼ¹²))
end


function χ(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ, op)
   (op((Sᵢ¹²*Sⱼ¹²), ρ¹²*Sₖ¹²)) * inv(0.5*(ρ-S)+Sₖ) 
end

χₖ⁺(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ) = χ(ρ, S, ρ¹², Sᵢ¹², Sⱼ¹², Sₖ¹², Sₖ,+)

function B(χₖ⁺,χₖ⁻, χᵢ⁺, χⱼ⁻)
   denom = (1.0 - χₖ⁻()*χₖ⁺())
   println(χₖ⁺.order)
   (χⱼ⁻() - χᵢ⁺()*χₖ⁺()) / denom , (χᵢ⁺() - χⱼ⁻()*χₖ⁻()) / denom
end

function v(𝐉ᵢ, 𝐉ⱼ, 𝐉ₖ, Bₖⁱʲ, χᵢ⁺, χⱼ⁻)
   (Bₖʲ, Bₖⁱ) = Bₖⁱʲ()
   0.25 * ((𝐉ₖ() - Bₖʲ*𝐉ᵢ() - Bₖⁱ*𝐉ⱼ()) / (1.0 - Bₖʲ*χᵢ⁺() - Bₖⁱ*χⱼ⁻()))
end