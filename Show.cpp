#include "Vocabulary.hpp"
#include "NTF.hpp"
#include "LBLM.hpp"
#include "Utils.hpp"
#include "Sigmoid.hpp"
#include <iostream>
#include <omp.h>

void NTF::SVOdist(Vocabulary& voc, int k){
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

    if (!voc.verbIndex.count(v) || (!voc.nounIndex.count(s) && s != voc.null) || !voc.nounIndex.count(o)){
      printf("retry\n~\n");
      continue;
    }

    MatD svo;
    
    if (s == voc.null){
      svo = this->verbMatrix[voc.verbIndex.at(v)]*this->nounVector.col(voc.nounIndex.at(o));
    }
    else {
      svo = this->nounVector.col(voc.nounIndex.at(s)).array()*(this->verbMatrix[voc.verbIndex.at(v)]*this->nounVector.col(voc.nounIndex.at(o))).array();
    }

    MatD dist(voc.svoListUniq.size(), 1);

#pragma omp parallel for num_threads(12)
    for (int i = 0; i < (int)voc.svoListUniq.size(); ++i){
      MatD tmp = this->nounVector.col(voc.svoListUniq[i]->s).array()*(this->verbMatrix[voc.svoListUniq[i]->v]*this->nounVector.col(voc.svoListUniq[i]->o)).array();

      dist.coeffRef(i, 0) = Utils::cosDis(svo, tmp);
    }

    std::vector<std::pair<std::string, double> > best, worst;

    for (int i = 0; i < k; ++i){
      int row, col;
      std::string str;

      dist.maxCoeff(&row, &col);
      str = voc.nounStr[voc.svoListUniq[row]->s]+" "+voc.verbStr[voc.svoListUniq[row]->v]+" "+voc.nounStr[voc.svoListUniq[row]->o];
      best.push_back(std::pair<std::string, double>(str, dist.coeff(row, col)));
      dist.coeffRef(row, col) = 0.0;
    }

    for (int i = 0; i < k/2; ++i){
      int row, col;
      std::string str;

      dist.minCoeff(&row, &col);
      str = voc.nounStr[voc.svoListUniq[row]->s]+" "+voc.verbStr[voc.svoListUniq[row]->v]+" "+voc.nounStr[voc.svoListUniq[row]->o];
      worst.push_back(std::pair<std::string, double>(str, dist.coeff(row, col)));
      dist.coeffRef(row, col) = 0.0;
    }

    for (int i = 0; i < (int)best.size(); ++i){
      printf("%s:%f\n", best[i].first.c_str(), best[i].second);
    }
    for (int i = (int)worst.size()-1; i >= 0; --i){
      printf("%s:%f\n", worst[i].first.c_str(), worst[i].second);
    }

    printf("~\n");
  }
}

void NTF::VOdist(Vocabulary& voc, int k){
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

    MatD vo = this->verbMatrix[voc.verbIndex.at(v)]*this->nounVector.col(voc.nounIndex.at(o));
    MatD dist(voc.svoListUniq.size(), 1);

    for (int i = 0; i < (int)voc.svoListUniq.size(); ++i){
      MatD tmp = this->verbMatrix[voc.svoListUniq[i]->v]*this->nounVector.col(voc.svoListUniq[i]->o);

      dist.coeffRef(i, 0) = Utils::cosDis(vo, tmp);
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

void NTF::SVO_PNdist(Vocabulary& voc, int k){
  printf("kNN SVOs\n");

  std::vector<std::string> res;

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    Utils::split(line, res);

    if (res.size() != 5){
      continue;
    }

    std::string s = res[0];
    std::string v = res[1];
    std::string o = res[2];
    std::string p = res[3];
    std::string n = res[4];

    if (!voc.verbIndex.count(v) || !voc.nounIndex.count(s) || !voc.nounIndex.count(o) || !voc.prepIndex.count(p) || !voc.nounIndex.count(n)){
      continue;
    }

    MatD svo = this->nounVector.col(voc.nounIndex.at(s)).array()*(this->verbMatrix[voc.verbIndex.at(v)]*this->nounVector.col(voc.nounIndex.at(o))).array();
    MatD svo_pn = svo.array()*(this->prepMatrix[voc.prepIndex.at(p)]*this->nounVector.col(voc.nounIndex.at(n))).array();
    MatD dist(voc.svopnListUniq.size(), 1);

    for (int i = 0; i < (int)voc.svopnListUniq.size(); ++i){
      MatD tmp = this->nounVector.col(voc.svopnListUniq[i]->svo->s).array()*(this->verbMatrix[voc.svopnListUniq[i]->svo->v]*this->nounVector.col(voc.svopnListUniq[i]->svo->o)).array();
      MatD tmp2 = tmp.array()*(this->prepMatrix[voc.svopnListUniq[i]->p]*this->nounVector.col(voc.svopnListUniq[i]->n)).array();

      dist.coeffRef(i, 0) = Utils::cosDis(svo_pn, tmp2);
    }

    for (int i = 0; i < k; ++i){
      int row, col;
      std::string str;

      dist.maxCoeff(&row, &col);
      str = voc.nounStr[voc.svopnListUniq[row]->svo->s]+" "+voc.verbStr[voc.svopnListUniq[row]->svo->v]+" "+voc.nounStr[voc.svopnListUniq[row]->svo->o]+" "+voc.prepStr[voc.svopnListUniq[row]->p]+" "+voc.nounStr[voc.svopnListUniq[row]->n];
      printf("%2d: %s (%f)\n", i+1, str.c_str(), dist.coeff(row, col));
      dist.coeffRef(row, col) = -1.0;
    }
  }
}

void NTF::SVOdist(Vocabulary& voc){
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

    if (!voc.verbIndex.count(v1) || !voc.verbIndex.count(v2) || (!voc.nounIndex.count(s1) && s1 != voc.null) || (!voc.nounIndex.count(s2) && s2 != voc.null) || !voc.nounIndex.count(o1) || !voc.nounIndex.count(o2)){
      continue;
    }

    MatD svo1, svo2;

    if (s1 == voc.null){
      svo1 =  this->verbMatrix[voc.verbIndex.at(v1)]*this->nounVector.col(voc.nounIndex.at(o1));
    }
    else {
      svo1 = this->nounVector.col(voc.nounIndex.at(s1)).array()*(this->verbMatrix[voc.verbIndex.at(v1)]*this->nounVector.col(voc.nounIndex.at(o1))).array();
    }

    if (s2 == voc.null){
      svo2 =  this->verbMatrix[voc.verbIndex.at(v2)]*this->nounVector.col(voc.nounIndex.at(o2));
    }
    else {
      svo2 = this->nounVector.col(voc.nounIndex.at(s2)).array()*(this->verbMatrix[voc.verbIndex.at(v2)]*this->nounVector.col(voc.nounIndex.at(o2))).array();
    }

    printf("%f\n", Utils::cosDis(svo1, svo2));
    if (s1 != voc.null && s2 != voc.null){
      printf("%s, %s: %f\n", s1.c_str(), s2.c_str(), Utils::cosDis(this->nounVector.col(voc.nounIndex.at(s1)), this->nounVector.col(voc.nounIndex.at(s2))));
    }
    printf("%s, %s: %f\n", v1.c_str(), v2.c_str(), Utils::cosDis(this->verbMatrix[voc.verbIndex.at(v1)], this->verbMatrix[voc.verbIndex.at(v2)]));
    printf("%s, %s: %f\n\n", o1.c_str(), o2.c_str(), Utils::cosDis(this->nounVector.col(voc.nounIndex.at(o1)), this->nounVector.col(voc.nounIndex.at(o2))));
  }
}

void NTF::vRowKnn(Vocabulary& voc, int k){
  printf("Verb Knn for rows\n");

  std::vector<std::string> res;

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    Utils::split(line, res);

    if (res.size() != 2){
      continue;
    }

    std::string v = res[0];
    int r = atoi(res[1].c_str());

    if (!voc.verbIndex.count(v) || r >= this->verbMatrix[0].rows()){
      continue;
    }

    MatD dist(1, this->verbMatrix.size());

    if (r < 0){
      MatD target = this->verbMatrix[voc.verbIndex.at(v)];

      for (int i = 0; i < dist.cols(); ++i){
	dist.coeffRef(0, i) = Utils::cosDis(target, this->verbMatrix[i]);
      }
    }
    else {
      MatD target = this->verbMatrix[voc.verbIndex.at(v)].row(r).transpose();

      for (int i = 0; i < dist.cols(); ++i){
	dist.coeffRef(0, i) = Utils::cosDis(target, this->verbMatrix[i].row(r).transpose());
      }    
    }

    for (int i = 0; i < k; ++i){
      int row, col;

      dist.maxCoeff(&row, &col);
      printf("(%.5f) %s\n", dist.coeff(row, col), voc.verbStr[col].c_str());
      dist.coeffRef(row, col) = -1.0;
    }

    printf("\n");
  }
}

void NTF::vColKnn(Vocabulary& voc, int k){
  printf("Verb Knn for cols\n");

  std::vector<std::string> res;

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    Utils::split(line, res);

    if (res.size() != 2){
      continue;
    }

    std::string v = res[0];
    int r = atoi(res[1].c_str());

    if (!voc.verbIndex.count(v) || r >= this->verbMatrix[0].cols()){
      continue;
    }

    MatD dist(1, this->verbMatrix.size());

    if (r < 0){
      MatD target = this->verbMatrix[voc.verbIndex.at(v)];

      for (int i = 0; i < dist.cols(); ++i){
	dist.coeffRef(0, i) = Utils::cosDis(target, this->verbMatrix[i]);
      }
    }
    else {
      MatD target = this->verbMatrix[voc.verbIndex.at(v)].col(r);

      for (int i = 0; i < dist.cols(); ++i){
	dist.coeffRef(0, i) = Utils::cosDis(target, this->verbMatrix[i].col(r));
      }    
    }

    for (int i = 0; i < k; ++i){
      int row, col;

      dist.maxCoeff(&row, &col);
      printf("(%.5f) %s\n", dist.coeff(row, col), voc.verbStr[col].c_str());
      dist.coeffRef(row, col) = -1.0;
    }

    printf("\n");
  }
}

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
