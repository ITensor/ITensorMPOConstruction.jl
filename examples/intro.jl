using ITensorMPOConstruction
using ITensors

function foo(N::Int; useITensorsAlg::Bool=false)::MPO
  os = OpSum{Float64}()
  @time "Constructing OpSum" for i in 1:N
    for j in (i + 1):N
      for k in (j + 1):N
        for l in (k + 1):N
          os .+= 1, "X", i, "X", j, "X", k, "X", l
          os .+= 1, "Y", i, "Y", j, "Y", k, "Y", l
          os .+= 1, "Z", i, "Z", j, "Z", k, "Z", l
        end
      end
    end
  end

  sites = siteinds("Qubit", N)

  if useITensorsAlg
    return @time "Constructing MPO" MPO(os, sites)
  else
    return @time "Constructing MPO" MPO_new(os, sites)
  end
end

let
  for N in 5:5:100
    println("N = $N")
    H = foo(N; useITensorsAlg=false)
    @show maxlinkdim(H)
    println()
  end
end

nothing;
