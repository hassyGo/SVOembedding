#pragma once

#include "Data.hpp"
#include "SVO.hpp"

class SVOPN : public Data{
public:
  int p, n;
  SVO* svo;

  int *p_, *n_;
  SVO **svo_;
};
