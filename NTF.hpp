#pragma once

#include "SVO.hpp"
#include "SVOPN.hpp"
#include "Vocabulary.hpp"
#include "Utils.hpp"
#include "Sigmoid.hpp"

class NTF{
public:
  MatD nounVector, nvG;
  std::vector<MatD> verbMatrix, vmG;
  std::vector<MatD> prepMatrix, pmG;

  MatD nounGrad;
  std::vector<MatD> verbGrad, prepGrad;
  std::unordered_map<int, int> nounMap, verbMap, prepMap;

  int negS, negV, negO; //for gradient checking
  int negP, negN; SVO* negSVO; //for gradient checking

  void init(const int dim, Vocabulary& voc);

  double score(const int s, const int v, const int o);
  double objective(SVO* svo);
  double score(const int p, const int n, SVO* svo);
  double objective(SVOPN* svopn);

  void train(std::vector<Data*>& sample, std::vector<Data*>& type, std::vector<Data*>& dummy, Vocabulary& voc, const double learningRate, const int maxItr, const int miniBatchSize, const int numNeg);
  void trainSVO(SVO* svo, Vocabulary& voc);
  void trainSVOPN(SVOPN* svopn, Vocabulary& voc);
  void update(const double learningRate, const int exception = -1);

  void gradCheck(SVO* svo);
  void gradCheck(SVOPN* svopn);

  void save(const std::string& file);
  void load(const std::string& file);

  double testSVO(Vocabulary& voc, const std::string& type, const bool ave = false);
  double testVO(Vocabulary& voc, const bool ave = false);
  void SVOdist(Vocabulary& voc);
  void SVOdist(Vocabulary& voc, int k);
  void VOdist(Vocabulary& voc, int k);
  void SVO_PNdist(Vocabulary& voc, int k);
  void vRowKnn(Vocabulary& voc, int k);
  void vColKnn(Vocabulary& voc, int k);
};

inline double NTF::score(const int s, const int v, const int o){
  return
    this->nounVector.col(s).transpose()*
    this->verbMatrix[v]*
    this->nounVector.col(o);
}

inline double NTF::objective(SVO* svo){
  return
    -log(Sigmoid::sigmoid(this->score(svo->s, svo->v, svo->o)))-
    log(1.0-Sigmoid::sigmoid(this->score(this->negS, svo->v, svo->o)))-
    log(1.0-Sigmoid::sigmoid(this->score(svo->s, this->negV, svo->o)))-
    //log(1.0-Sigmoid::sigmoid(this->score(this->negS, svo->v, this->negO)))+
    log(1.0-Sigmoid::sigmoid(this->score(svo->s, svo->v, this->negO)));
}

inline double NTF::score(const int p, const int n, SVO* svo){
  MatD SVO = this->nounVector.col(svo->s).array()*(this->verbMatrix[svo->v]*this->nounVector.col(svo->o)).array();

  return
    SVO.col(0).transpose()*
    this->prepMatrix[p]*
    this->nounVector.col(n);
}

inline double NTF::objective(SVOPN* svopn){
  return
    -log(Sigmoid::sigmoid(this->score(svopn->p, svopn->n, svopn->svo)))-
    log(1.0-Sigmoid::sigmoid(this->score(this->negP, svopn->n, svopn->svo)))-
    log(1.0-Sigmoid::sigmoid(this->score(svopn->p, this->negN, svopn->svo)))-
    log(1.0-Sigmoid::sigmoid(this->score(svopn->p, svopn->n, this->negSVO)));
}
