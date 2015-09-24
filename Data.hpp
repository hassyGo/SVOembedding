#pragma once

#include <vector>
#include "Rand.hpp"

class Vocabulary;

class Data{
public:
  enum TYPE{
    SVO_,
    SVOPN_,
  };

  enum SET{
    TRAIN,
    DEV,
    TEST,
  };

  static Rand rndModel;
  static Rand rndData;

  Data::TYPE type;
  Data::SET set;
};
