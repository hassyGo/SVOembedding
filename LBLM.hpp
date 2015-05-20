#pragma once

#include "SVO.hpp"
#include "SVOPN.hpp"
#include "Vocabulary.hpp"
#include "Tanh.hpp"
#include "Utils.hpp"

class LBLM{
public:
  enum TARGET{
    S, V, O,
    P, N,
  };

  //word vectors
  MatD nounVector;
  MatD verbVector;
  MatD prepVector;

  //composition weight vectors for SVO
  MatD subjCompWeight;
  MatD verbCompWeight;
  MatD objCompWeight;

  //context weight vectors for SVO
  MatD subjWeight;
  MatD verbWeight;
  MatD objWeight;

  //context weight vectors for SVO-P-N
  MatD prepWeight;
  MatD pobjWeight;
  MatD svoWeight;

  //scoring weight vectors
  MatD nounScoreWeight;
  MatD verbScoreWeight;
  MatD prepScoreWeight;

  MatD nounGrad, nounGradHist;
  MatD verbGrad, verbGradHist;
  MatD prepGrad, prepGradHist;
  MatD scwGrad, vcwGrad, ocwGrad;
  MatD scwGradHist, vcwGradHist, ocwGradHist;
  MatD swGrad, vwGrad, owGrad;
  MatD swGradHist, vwGradHist, owGradHist;
  MatD pwGrad, powGrad, svowGrad;
  MatD pwGradHist, powGradHist, svowGradHist;
  MatD nounScoreGrad, nounScoreGradHist;
  MatD verbScoreGrad, verbScoreGradHist;
  MatD prepScoreGrad, prepScoreGradHist;

  std::unordered_map<int, int> nounMap;
  std::unordered_map<int, int> verbMap;
  std::unordered_map<int, int> prepMap;
  std::unordered_map<int, int> nounScoreMap;
  std::unordered_map<int, int> verbScoreMap;
  std::unordered_map<int, int> prepScoreMap;

  int negS, negV, negO;
  int negP, negN;

  void init(const int dim, Vocabulary& voc);

  void compose(MatD& comp, SVO* svo);
  void train(std::vector<Data*>& instance, std::vector<Data*>& type, std::vector<Data*>& dummy, Vocabulary& voc, const double learningRate, const int maxItr, const int miniBatchSize, const int numNeg);
  void train(SVO* svo, Vocabulary& voc);
  void train(SVOPN* svopn, Vocabulary& voc);
  void update(const double learningRate, const int exception = -1);

  double score(const int s, const int v, const int o, const LBLM::TARGET target);
  double objective(SVO* svo);
  double score(const int p, const int n, SVO* svo, const LBLM::TARGET target);
  void gradCheck(SVO* svo);
  void gradCheck(SVOPN* svopn);

  void SVOdist(Vocabulary& voc);
  void SVOdist(Vocabulary& voc, int k);
  void VOdist(Vocabulary& voc, int k);

  void save(const std::string& file);
  void load(const std::string& file);
};

inline double LBLM::score(const int s, const int v, const int o, const LBLM::TARGET target){
  MatD f;

  if (target == LBLM::S){
    f = this->objWeight.array()*this->nounVector.col(o).array()+this->verbWeight.array()*this->verbVector.col(v).array();
    Tanh::tanh(f);
    return (this->nounScoreWeight.row(s)*f).coeff(0, 0);
  }
  else if (target == LBLM::V){
    f = this->subjWeight.array()*this->nounVector.col(s).array()+this->objWeight.array()*this->nounVector.col(o).array();
    Tanh::tanh(f);
    return (this->verbScoreWeight.row(v)*f).coeff(0, 0);
  }
  else if (target == LBLM::O){
    f = this->subjWeight.array()*this->nounVector.col(s).array()+this->verbWeight.array()*this->verbVector.col(v).array();
    Tanh::tanh(f);
    return (this->nounScoreWeight.row(o)*f).coeff(0, 0);
  }

  return -1.0e+10;
}

inline double LBLM::objective(SVO* svo){
  return
    Utils::max(1.0+this->score(this->negS, svo->v, svo->o, LBLM::S)-this->score(svo->s, svo->v, svo->o, LBLM::S), 0.0)+
    Utils::max(1.0+this->score(svo->s, this->negV, svo->o, LBLM::V)-this->score(svo->s, svo->v, svo->o, LBLM::V), 0.0)+
    Utils::max(1.0+this->score(svo->s, svo->v, this->negO, LBLM::O)-this->score(svo->s, svo->v, svo->o, LBLM::O), 0.0);
}

inline double LBLM::score(const int p, const int n, SVO* svo, const LBLM::TARGET target){
  MatD f;
  MatD svoVec;

  this->compose(svoVec, svo);

  if (target == LBLM::P){
    f = this->pobjWeight.array()*this->nounVector.col(n).array()+this->svoWeight.array()*svoVec.array();
    Tanh::tanh(f);
    return (this->prepScoreWeight.row(p)*f).coeff(0, 0);
  }
  else if (target == LBLM::N){
    f = this->prepWeight.array()*this->prepVector.col(p).array()+this->svoWeight.array()*svoVec.array();
    Tanh::tanh(f);
    return (this->nounScoreWeight.row(n)*f).coeff(0, 0);
  }

  return -1.0e+10;
}

inline void LBLM::compose(MatD& comp, SVO* svo){
  if (svo->s == -1){
    comp =
      this->verbCompWeight.array()*this->verbVector.col(svo->v).array()+
      this->objCompWeight.array()*this->nounVector.col(svo->o).array();
  }
  else {
    comp =
      this->subjCompWeight.array()*this->nounVector.col(svo->s).array()+
      this->verbCompWeight.array()*this->verbVector.col(svo->v).array()+
      this->objCompWeight.array()*this->nounVector.col(svo->o).array();
  }

  Tanh::tanh(comp);
}
