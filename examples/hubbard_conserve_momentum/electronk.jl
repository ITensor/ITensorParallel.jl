import ITensors: space, state, op, has_fermion_string

function space(
  ::SiteType"ElectronK",
  n::Int;
  conserve_qns=false,
  conserve_sz=conserve_qns,
  conserve_nf=conserve_qns,
  conserve_nfparity=conserve_qns,
  conserve_ky=false,
  qnname_sz="Sz",
  qnname_nf="Nf",
  qnname_nfparity="NfParity",
  qnname_ky="Ky",
  modulus_ky=nothing,
  # Deprecated
  conserve_parity=nothing,
)
  if !isnothing(conserve_parity)
    conserve_nfparity = conserve_parity
  end
  if conserve_ky && conserve_sz && conserve_nf
    mod = (n - 1) % modulus_ky
    mod2 = (2 * mod) % modulus_ky
    return [
      QN((qnname_nf, 0, -1), (qnname_sz, 0), (qnname_ky, 0, modulus_ky)) => 1
      QN((qnname_nf, 1, -1), (qnname_sz, 1), (qnname_ky, mod, modulus_ky)) => 1
      QN((qnname_nf, 1, -1), (qnname_sz, -1), (qnname_ky, mod, modulus_ky)) => 1
      QN((qnname_nf, 2, -1), (qnname_sz, 0), (qnname_ky, mod2, modulus_ky)) => 1
    ]
  elseif conserve_ky
    error("Cannot conserve ky without conserving sz and nf")
  elseif conserve_sz && conserve_nf
    return [
      QN((qnname_nf, 0, -1), (qnname_sz, 0)) => 1
      QN((qnname_nf, 1, -1), (qnname_sz, +1)) => 1
      QN((qnname_nf, 1, -1), (qnname_sz, -1)) => 1
      QN((qnname_nf, 2, -1), (qnname_sz, 0)) => 1
    ]
  elseif conserve_nf
    return [
      QN(qnname_nf, 0, -1) => 1
      QN(qnname_nf, 1, -1) => 2
      QN(qnname_nf, 2, -1) => 1
    ]
  elseif conserve_sz
    return [
      QN((qnname_sz, 0), (qnname_nfparity, 0, -2)) => 1
      QN((qnname_sz, +1), (qnname_nfparity, 1, -2)) => 1
      QN((qnname_sz, -1), (qnname_nfparity, 1, -2)) => 1
      QN((qnname_sz, 0), (qnname_nfparity, 0, -2)) => 1
    ]
  elseif conserve_nfparity
    return [
      QN(qnname_nfparity, 0, -2) => 1
      QN(qnname_nfparity, 1, -2) => 2
      QN(qnname_nfparity, 0, -2) => 1
    ]
  end
  return 4
end

state(statename::StateName, ::SiteType"ElectronK") = state(statename, SiteType("Electron"))
op(opname::OpName, ::SiteType"ElectronK") = op(opname, SiteType("Electron"))
has_fermion_string(opname::OpName, ::SiteType"ElectronK") = has_fermion_string(opname, SiteType("Electron"))
