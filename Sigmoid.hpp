#pragma once

#include "Matrix.hpp"

class Sigmoid{
public:
  static double sigmoid(const double x);
  static double sigmoid(const MatD& x);
  static void sigmoid(MatD& x);
};

inline double Sigmoid::sigmoid(const double x){
  return 1.0/(1.0+::exp(-x));
}

inline double Sigmoid::sigmoid(const MatD& x){
  return 1.0/(1.0+::exp(-x.coeff(0, 0)));
}

inline void Sigmoid::sigmoid(MatD& x){
  x = x.unaryExpr(std::ptr_fun((double (*)(const double))Sigmoid::sigmoid));
}
