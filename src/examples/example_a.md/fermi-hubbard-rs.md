# Fermi-Hubbard Hamiltonian in Real Space
The one dimensional Fermi-Hubbard Hamiltonian with periodic boundary conditions on ``N`` sites can be expressed in real space as

``
\mathcal{H} = -t \sum_{i = 1}^N \sum_{\sigma \in (\uparrow, \downarrow)} \left( c^\dagger_{i, \sigma} c_{i + 1, \sigma} + c^\dagger_{i, \sigma} c_{i - 1, \sigma} \right) + U \sum_{i = 1}^N n_{i, \uparrow} n_{i, \downarrow} \ ,
``

where the periodic boundary conditions enforce that ``c_k = c_{k + N}``. For this Hamiltonian all that needs to be done to switch over to using ITensorMPOConstruction is switch `MPO(os, sites)` to `MPO_New(os, sites)`.

````julia
using ITensors, ITensorMPS, ITensorMPOConstruction

function Fermi_Hubbard_real_space(
  N::Int, t::Real=1.0, U::Real=4.0; useITensorsAlg::Bool=false
)::MPO
  os = OpSum{Float64}()
  for i in 1:N
    for j in [mod1(i + 1, N), mod1(i - 1, N)]
      os .+= -t, "Cdagup", i, "Cup", j
      os .+= -t, "Cdagdn", i, "Cdn", j
    end

    os .+= U, "Nup * Ndn", i
  end

  sites = siteinds("Electron", N; conserve_qns=true)

  if useITensorsAlg
    return MPO(os, sites)
  else
    return MPO_new(os, sites)
  end
end
````

For ``N = 1000`` both ITensorMPS and ITensorMPOConstruction can construct an MPO of bond dimension 10 in under two seconds. When quantum number conservation is enabled, ITensorMPS produces an MPO that is 93.4% block sparse, whereas ITensorMPOConstruction's MPO is 97.4% block sparse.

````julia
let N = 1000
  for useITensorsAlg in [true, false]
      alg = useITensorsAlg ? "ITensorMPS" : "ITensorMPOConstruction"

      println("Constructing the Fermi-Hubbard real space MPO for $N sites using $alg")
      @time "Total construction time" mpo = Fermi_Hubbard_real_space(
        N; useITensorsAlg=useITensorsAlg
      )
      println("The maximum bond dimension is $(maxlinkdim(mpo))")
      println("The sparsity is $(ITensorMPOConstruction.sparsity(mpo))")
      println()
  end
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

