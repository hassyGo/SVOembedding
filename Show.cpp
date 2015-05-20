#include "Vocabulary.hpp"
#include "LBLM.hpp"
#include "Utils.hpp"
#include "Sigmoid.hpp"
#include <iostream>
#include <omp.h>

void LBLM::SVOdist(Vocabulary& voc, int k){
  printf("kNN SVOs\n");

  std::vector<std::string> res;

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    Utils::split(line, res);

    if (res.size() != 3){
      continue;
    }

    std::string s = res[0];
    std::string v = res[1];
    std::string o = res[2];

    if (!voc.verbIndex.count(v) || !voc.nounIndex.count(s) || !voc.nounIndex.count(o)){
      continue;
    }

    SVO svoInst;
    MatD svo, svoTmp;

    svoInst.s = voc.nounIndex.at(s); svoInst.v = voc.verbIndex.at(v); svoInst.o = voc.nounIndex.at(o);
    this->compose(svo, &svoInst);

    MatD dist(voc.svoListUniq.size(), 1);

    for (int i = 0; i < (int)voc.svoListUniq.size(); ++i){
      this->compose(svoTmp, voc.svoListUniq[i]);

      dist.coeffRef(i, 0) = Utils::cosDis(svo, svoTmp);
    }

    for (int i = 0; i < k; ++i){
      int row, col;
      std::string str;

      dist.maxCoeff(&row, &col);
      str = voc.nounStr[voc.svoListUniq[row]->s]+" "+voc.verbStr[voc.svoListUniq[row]->v]+" "+voc.nounStr[voc.svoListUniq[row]->o];
      printf("%2d: %s (%f)\n", i+1, str.c_str(), dist.coeff(row, col));
      dist.coeffRef(row, col) = -1.0;
    }
  }
}

void LBLM::VOdist(Vocabulary& voc, int k){
  printf("kNN VOs\n");

  std::vector<std::string> res;

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    Utils::split(line, res);

    if (res.size() != 2){
      continue;
    }

    std::string v = res[0];
    std::string o = res[1];

    if (!voc.verbIndex.count(v) || !voc.nounIndex.count(o)){
      continue;
    }

    SVO svoInst;
    MatD vo, voTmp;
    MatD dist(voc.svoListUniq.size(), 1);

    svoInst.s = voc.nullIndex; svoInst.v = voc.verbIndex.at(v); svoInst.o = voc.nounIndex.at(o);
    this->compose(vo, &svoInst);

    for (int i = 0; i < (int)voc.svoListUniq.size(); ++i){
      svoInst.v = voc.svoListUniq[i]->v; svoInst.o = voc.svoListUniq[i]->o;
      this->compose(voTmp, &svoInst);
      dist.coeffRef(i, 0) = Utils::cosDis(vo, voTmp);
    }

    double prevScore = -1.0;

    for (int i = 0; i < k; ++i){
      int row, col;
      std::string str;

      dist.maxCoeff(&row, &col);

      if (dist.coeff(row, col) == prevScore){
	dist.coeffRef(row, col) = -1.0;
	--i;
	continue;
      }

      prevScore = dist.coeff(row, col);
      str = voc.verbStr[voc.svoListUniq[row]->v]+" "+voc.nounStr[voc.svoListUniq[row]->o];
      printf("%2d: %s (%f)\n", i+1, str.c_str(), dist.coeff(row, col));
      dist.coeffRef(row, col) = -1.0;
    }
  }
}

void LBLM::SVOdist(Vocabulary& voc){
  printf("Distance between SVOs\n");

  std::vector<std::string> res;

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    Utils::split(line, res);

    if (res.size() != 6){
      continue;
    }

    std::string s1 = res[0];
    std::string v1 = res[1];
    std::string o1 = res[2];
    std::string s2 = res[3];
    std::string v2 = res[4];
    std::string o2 = res[5];

    if (!voc.verbIndex.count(v1) || !voc.verbIndex.count(v2) || !voc.nounIndex.count(s1) || !voc.nounIndex.count(s2) || !voc.nounIndex.count(o1) || !voc.nounIndex.count(o2)){
      continue;
    }

    SVO svo;
    MatD svo1, svo2;

    svo.s = voc.nounIndex.at(s1); svo.v = voc.verbIndex.at(v1); svo.o = voc.nounIndex.at(o1);
    this->compose(svo1, &svo);
    svo.s = voc.nounIndex.at(s2); svo.v = voc.verbIndex.at(v2); svo.o = voc.nounIndex.at(o2);
    this->compose(svo2, &svo);

    printf("%f\n", Utils::cosDis(svo1, svo2));
    printf("%s, %s: %f\n", s1.c_str(), s2.c_str(), Utils::cosDis(this->nounVector.col(voc.nounIndex.at(s1)), this->nounVector.col(voc.nounIndex.at(s2))));
    printf("%s, %s: %f\n", v1.c_str(), v2.c_str(), Utils::cosDis(this->verbVector.col(voc.verbIndex.at(v1)), this->verbVector.col(voc.verbIndex.at(v2))));
    printf("%s, %s: %f\n\n", o1.c_str(), o2.c_str(), Utils::cosDis(this->nounVector.col(voc.nounIndex.at(o1)), this->nounVector.col(voc.nounIndex.at(o2))));
  }
}
