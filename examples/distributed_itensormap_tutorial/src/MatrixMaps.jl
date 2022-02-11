# Wrapper of Matrix that is callable as `A(v)`
struct MatrixMap{T}
  data::Matrix{T}
end
(A::MatrixMap)(v) = A.data * v
Base.:+(A::MatrixMap, B::MatrixMap) = MatrixMap(A.data + B.data)
