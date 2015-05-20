#pragma once

#include "Matrix.hpp"

class Tanh{
public:
  static void tanh(MatD& x);
  static MatD tanhPrime(const MatD& x);
};

//f(x) = tanh(x)
inline void Tanh::tanh(MatD& x){
  x = x.unaryExpr(std::ptr_fun(::tanh));
}

//x must be a output of Tanh::tanh
//f'(x) = 1-(f(x))^2
inline MatD Tanh::tanhPrime(const MatD& x){
  return 1.0-x.array().square();
}
