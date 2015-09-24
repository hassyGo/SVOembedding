#include "NTF.hpp"
#include "Sigmoid.hpp"
#include "Utils.hpp"
#include <fstream>
#include <iostream>

#define ETA 1.0

void NTF::init(const int dim, Vocabulary& voc){
  const double scale1 = 1.0/dim;
  const double scale2 = 1.0/(dim*dim);

  this->nounVector = MatD(dim, voc.nounStr.size());
  Data::rndModel.gauss(this->nounVector, scale1);

  if (voc.nullIndex >= 0){
    this->nounVector.col(voc.nullIndex).fill(1.0);
  }

  for (int i = 0; i < (int)voc.verbStr.size(); ++i){
    this->verbMatrix.push_back(MatD(dim, dim));
    Data::rndModel.gauss(this->verbMatrix.back(), scale2);
  }
  for (int i = 0; i < (int)voc.prepStr.size(); ++i){
    this->prepMatrix.push_back(MatD(dim, dim));
    Data::rndModel.gauss(this->prepMatrix.back(), scale2);
  }

  //gradient
  this->nounGrad = MatD::Zero(this->nounVector.rows(), this->nounVector.cols());
  
  for (int i = 0; i < (int)this->verbMatrix.size(); ++i){
    this->verbGrad.push_back(MatD::Zero(this->verbMatrix[i].rows(), this->verbMatrix[i].cols()));
  }
  for (int i = 0; i < (int)this->prepMatrix.size(); ++i){
    this->prepGrad.push_back(MatD::Zero(this->prepMatrix[i].rows(), this->prepMatrix[i].cols()));
  }

  //AdaGrad
  this->nvG = MatD::Ones(this->nounVector.rows(), this->nounVector.cols())*ETA;

  for (int i = 0; i < (int)this->verbMatrix.size(); ++i){
    this->vmG.push_back(MatD::Ones(this->verbMatrix[i].rows(), this->verbMatrix[i].cols())*ETA);
  }
  for (int i = 0; i < (int)this->prepMatrix.size(); ++i){
    this->pmG.push_back(MatD::Ones(this->prepMatrix[i].rows(), this->prepMatrix[i].cols())*ETA);
  }
}

void NTF::train(std::vector<Data*>& sample, std::vector<Data*>& type, std::vector<Data*>& dummy, Vocabulary& voc, const double learningRate, const int maxItr, const int miniBatchSize, const int numNeg){
  const int prog = sample.size()/10;
  int counter = 0;

  for (int itr = 0; itr < maxItr; ++itr){
    counter = 0;
    std::random_shuffle(sample.begin(), sample.end());
    ::printf("### NTF: itr %2d/%2d, learning rate %.6f, dim %d\n", itr+1, maxItr, learningRate, (int)this->nounVector.rows());

    for (int i = 0; i < (int)sample.size(); ++i){
      if (sample[i]->type == Data::SVO_){
	this->trainSVO((SVO*)sample[i], voc);
	//gradCheck((SVO::SVO*)sample[i]);
      }
      else {
	this->trainSVOPN((SVOPN*)sample[i], voc);
	//gradCheck((SVOPN*)sample[i]);
      }

      if ((i+1)%miniBatchSize == 0 || i == (int)sample.size()-1){
	this->update(learningRate, voc.nullIndex);
      }
      
      if ((i+1)%prog == 0){
	std::cout << (++counter)*10 << "% " << std::flush;
      }
    }

    std::cout << std::endl;

    MatD accDev = MatD::Zero(numNeg, 1);
    double devCount = 0.0;

    for (int j = 0; j < (int)type.size(); ++j){
      if (type[j]->type == Data::SVO_){
	SVO* sample = (SVO*)type[j];

	if (sample->set == Data::TEST){
	  devCount += 1.0;
	}
	else {
	  continue;
	}

	double svo = this->score(sample->s, sample->v, sample->o);

	for (int k = 0; k < accDev.rows(); ++k){
	  double s_vo = this->score(sample->s_[k], sample->v, sample->o);
	  double svo_ = this->score(sample->s, sample->v, sample->o_[k]);
	  double sv_o = this->score(sample->s, sample->v_[k], sample->o);
	  
	  if (svo > s_vo && svo > svo_ && svo > sv_o){
	    accDev.coeffRef(k, 0) += 1.0;
	  }
	}
      }
      else if (type[j]->type == Data::SVOPN_){
	SVOPN* sample = (SVOPN*)type[j];

	if (sample->set == Data::TEST){
	  devCount += 1.0;
	}
	else {
	  continue;
	}

	double svoPn = this->score(sample->p, sample->n, sample->svo);

	for (int k = 0; k < accDev.rows(); ++k){
	  double svo_Pn = this->score(sample->p, sample->n, sample->svo_[k]);
	  double svoPn_ = this->score(sample->p, sample->n_[k], sample->svo);
	  double svoP_n = this->score(sample->p_[k], sample->n, sample->svo);
	
	  if (svoPn > svo_Pn && svoPn > svoPn_ && svoPn > svoP_n){
	    accDev.coeffRef(k, 0) += 1.0;
	  }
	}
      }
    }

    accDev /= devCount;
    accDev *= 100.0;
    printf("\tDev Acc: %f (%f)", accDev.array().sum()/accDev.rows(), Utils::stdDev(accDev));
    std::cout << std::endl;
  }
}

void NTF::trainSVO(SVO* svo, Vocabulary& voc){
  static double deltaPos, deltaNeg;
  static MatD vo, _vo, so, vs;

  this->verbMap[svo->v] = 1;
  this->nounMap[svo->s] = 1;
  this->nounMap[svo->o] = 1;
  vo = this->verbMatrix[svo->v]*this->nounVector.col(svo->o);
  so = this->nounVector.col(svo->s)*this->nounVector.col(svo->o).transpose();
  vs = this->verbMatrix[svo->v].transpose()*this->nounVector.col(svo->s);
  deltaPos = Sigmoid::sigmoid(this->nounVector.col(svo->s).transpose()*vo)-1.0;

  this->nounGrad.col(svo->s) += deltaPos*vo;
  this->nounGrad.col(svo->o) += deltaPos*vs;
  this->verbGrad[svo->v] += deltaPos*so;

  //V
  do {
    this->negV = voc.verbList[(Data::rndData.next() >> 16)%voc.verbList.size()];
  } while (voc.exist(svo->s, this->negV, svo->o));
  
  this->verbMap[this->negV] = 1;
    
  _vo = this->verbMatrix[this->negV]*this->nounVector.col(svo->o);
  deltaNeg = Sigmoid::sigmoid(this->nounVector.col(svo->s).transpose()*_vo);
  this->nounGrad.col(svo->s) += deltaNeg*_vo;
  this->nounGrad.col(svo->o) += deltaNeg*(this->verbMatrix[this->negV].transpose()*this->nounVector.col(svo->s));
  this->verbGrad[this->negV] += deltaNeg*so;
  
  //S
  do {
    this->negS = voc.vsubjList[(Data::rndData.next() >> 16)%voc.vsubjList.size()];
  } while (voc.exist(this->negS, svo->v, svo->o));

  this->nounMap[this->negS] = 1;

  deltaNeg = Sigmoid::sigmoid(this->nounVector.col(this->negS).transpose()*vo);
  this->nounGrad.col(svo->o) += deltaNeg*(this->verbMatrix[svo->v].transpose()*this->nounVector.col(this->negS));
  this->verbGrad[svo->v] += deltaNeg*(this->nounVector.col(this->negS)*this->nounVector.col(svo->o).transpose());
  this->nounGrad.col(this->negS) += deltaNeg*vo;

  //O
  do {
    this->negO = voc.vobjList[(Data::rndData.next() >> 16)%voc.vobjList.size()];
  } while (voc.exist(svo->s, svo->v, this->negO) || voc.exist(this->negS, svo->v, this->negO));

  this->nounMap[this->negO] = 1;
  
  _vo = this->verbMatrix[svo->v]*this->nounVector.col(this->negO);
  deltaNeg = Sigmoid::sigmoid(this->nounVector.col(svo->s).transpose()*_vo);
  this->nounGrad.col(svo->s) += deltaNeg*_vo;
  this->verbGrad[svo->v] += deltaNeg*(this->nounVector.col(svo->s)*this->nounVector.col(this->negO).transpose());
  this->nounGrad.col(this->negO) += deltaNeg*vs;
}

void NTF::trainSVOPN(SVOPN* svopn, Vocabulary& voc){
  static double deltaPos, deltaNeg;
  static MatD pn, _pn, psvo, svon;
  static MatD svo, svo_, vo, vo_, svoDelta, svoDelta_;

  this->prepMap[svopn->p] = 1;
  this->nounMap[svopn->n] = 1;
  this->verbMap[svopn->svo->v] = 1;
  this->nounMap[svopn->svo->s] = 1;
  this->nounMap[svopn->svo->o] = 1;

  vo = this->verbMatrix[svopn->svo->v]*this->nounVector.col(svopn->svo->o);
  svo =  this->nounVector.col(svopn->svo->s).array()*vo.array();
  pn = this->prepMatrix[svopn->p]*this->nounVector.col(svopn->n);
  psvo = this->prepMatrix[svopn->p].transpose()*svo;
  svon = svo*this->nounVector.col(svopn->n).transpose();
  deltaPos = Sigmoid::sigmoid(svo.transpose()*pn)-1.0;

  this->nounGrad.col(svopn->n) += deltaPos*psvo;
  this->prepGrad[svopn->p] += deltaPos*svon;

  svoDelta = deltaPos*pn;

  //P
  do {
    this->negP = voc.prepList[(Data::rndData.next() >> 16)%voc.prepList.size()];
  } while (voc.exist(this->negP, svopn->n, svopn->svo));

  this->prepMap[this->negP] = 1;

  _pn = this->prepMatrix[this->negP]*this->nounVector.col(svopn->n);
  deltaNeg = Sigmoid::sigmoid(svo.transpose()*_pn);
  this->nounGrad.col(svopn->n) += deltaNeg*(this->prepMatrix[this->negP].transpose()*svo);
  this->prepGrad[this->negP] += deltaNeg*svon;
  svoDelta += deltaNeg*_pn;

  //SVO
  do {
    this->negSVO = voc.svoList[(Data::rndData.next() >> 16)%voc.svoList.size()];
  } while (voc.exist(svopn->p, svopn->n, this->negSVO));

  this->verbMap[this->negSVO->v] = 1;
  this->nounMap[this->negSVO->s] = 1;
  this->nounMap[this->negSVO->o] = 1;

  vo_ = this->verbMatrix[this->negSVO->v]*this->nounVector.col(this->negSVO->o);
  svo_ =  this->nounVector.col(this->negSVO->s).array()*vo_.array();
  deltaNeg = Sigmoid::sigmoid(svo_.transpose()*pn);
  this->nounGrad.col(svopn->n) += deltaNeg*(this->prepMatrix[svopn->p].transpose()*svo_);
  this->prepGrad[svopn->p] += deltaNeg*(svo_*this->nounVector.col(svopn->n).transpose());
  svoDelta_ = deltaNeg*pn;

  //N
  do {
    this->negN = voc.pobjList[(Data::rndData.next() >> 16)%voc.pobjList.size()];
  } while (voc.exist(svopn->p, this->negN, svopn->svo));

  this->nounMap[this->negN] = 1;

  _pn = this->prepMatrix[svopn->p]*this->nounVector.col(this->negN);
  deltaNeg = Sigmoid::sigmoid(svo.transpose()*_pn);
  this->nounGrad.col(this->negN) += deltaNeg*psvo;
  this->prepGrad[svopn->p] += deltaNeg*(svo*this->nounVector.col(this->negN).transpose());
  svoDelta += deltaNeg*_pn;

  //pos
  this->nounGrad.col(svopn->svo->s).array() += svoDelta.array()*vo.array();
  svoDelta.array() *= this->nounVector.col(svopn->svo->s).array();
  this->verbGrad[svopn->svo->v] += svoDelta*this->nounVector.col(svopn->svo->o).transpose();
  this->nounGrad.col(svopn->svo->o) += this->verbMatrix[svopn->svo->v].transpose()*svoDelta;
  //neg
  this->nounGrad.col(this->negSVO->s).array() += svoDelta_.array()*vo_.array();
  svoDelta_.array() *= this->nounVector.col(this->negSVO->s).array();
  this->verbGrad[this->negSVO->v] += svoDelta_*this->nounVector.col(this->negSVO->o).transpose();
  this->nounGrad.col(this->negSVO->o) += this->verbMatrix[this->negSVO->v].transpose()*svoDelta_;
}


void NTF::update(const double learningRate, const int exception){
  for (std::unordered_map<int, int>::iterator it = this->nounMap.begin(); it != this->nounMap.end(); ++it){
    if (it->first == exception){
      this->nounGrad.col(it->first).setZero();
      continue;
    }

    //AdaGrad
    this->nvG.col(it->first).array() += this->nounGrad.col(it->first).array().square();
    this->nounGrad.col(it->first).array() /= this->nvG.col(it->first).array().sqrt();
    this->nounVector.col(it->first) -= learningRate*this->nounGrad.col(it->first);
    
    this->nounGrad.col(it->first).setZero();
  }

  for (std::unordered_map<int, int>::iterator it = this->verbMap.begin(); it != this->verbMap.end(); ++it){
    //AdaGrad
    this->vmG[it->first].array() += this->verbGrad[it->first].array().square();
    this->verbGrad[it->first].array() /= this->vmG[it->first].array().sqrt();
    this->verbMatrix[it->first] -= learningRate*this->verbGrad[it->first];

    this->verbGrad[it->first].setZero();
  }

  for (std::unordered_map<int, int>::iterator it = this->prepMap.begin(); it != this->prepMap.end(); ++it){
    //AdaGrad
    this->pmG[it->first].array() += this->prepGrad[it->first].array().square();
    this->prepGrad[it->first].array() /= this->pmG[it->first].array().sqrt();
    this->prepMatrix[it->first] -= learningRate*this->prepGrad[it->first];

    this->prepGrad[it->first].setZero();
  }

  this->nounMap.clear();
  this->verbMap.clear();
  this->prepMap.clear();
}

void NTF::gradCheck(SVO* svo){
  const double eps = 1.0e-04;
  double val, objPlus, objMinus;

  printf("\nchecking gradients ...\n");

  for (std::unordered_map<int, int>::iterator it = this->nounMap.begin(); it != this->nounMap.end(); ++it){
    printf("----------- noun %10d -------------\n", it->first);

    for (int i = 0; i < this->nounGrad.rows(); ++i){
      val = this->nounVector.coeff(i, it->first);
      this->nounVector.coeffRef(i, it->first) = val+eps;
      objPlus = this->objective(svo);
      this->nounVector.coeffRef(i, it->first) = val-eps;
      objMinus = this->objective(svo);
      this->nounVector.coeffRef(i, it->first) = val;
      printf("backprop:  %.8f\n", this->nounGrad.coeff(i, it->first));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
    }
  }

  for (std::unordered_map<int, int>::iterator it = this->verbMap.begin(); it != this->verbMap.end(); ++it){
    printf("----------- verb %10d -------------\n", it->first);

    for (int i = 0; i < this->verbGrad[it->first].rows(); ++i){
      for (int j = 0; j < this->verbGrad[it->first].cols(); ++j){
	val = this->verbMatrix[it->first].coeff(i, j);
	this->verbMatrix[it->first].coeffRef(i, j) = val+eps;
	objPlus = this->objective(svo);
	this->verbMatrix[it->first].coeffRef(i, j) = val-eps;
	objMinus = this->objective(svo);
	this->verbMatrix[it->first].coeffRef(i, j) = val;
	printf("backprop:  %.8f\n", this->verbGrad[it->first].coeff(i, j));
	printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
      }
    }
  }
}

void NTF::gradCheck(SVOPN* svopn){
  const double eps = 1.0e-04;
  double val, objPlus, objMinus;

  printf("\nchecking gradients ...\n");

  for (std::unordered_map<int, int>::iterator it = this->nounMap.begin(); it != this->nounMap.end(); ++it){
    printf("----------- noun %10d -------------\n", it->first);

    for (int i = 0; i < this->nounGrad.rows(); ++i){
      val = this->nounVector.coeff(i, it->first);
      this->nounVector.coeffRef(i, it->first) = val+eps;
      objPlus = this->objective(svopn);
      this->nounVector.coeffRef(i, it->first) = val-eps;
      objMinus = this->objective(svopn);
      this->nounVector.coeffRef(i, it->first) = val;
      printf("backprop:  %.8f\n", this->nounGrad.coeff(i, it->first));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
    }
  }

  for (std::unordered_map<int, int>::iterator it = this->verbMap.begin(); it != this->verbMap.end(); ++it){
    printf("----------- verb %10d -------------\n", it->first);

    for (int i = 0; i < this->verbGrad[it->first].rows(); ++i){
      for (int j = 0; j < this->verbGrad[it->first].cols(); ++j){
	val = this->verbMatrix[it->first].coeff(i, j);
	this->verbMatrix[it->first].coeffRef(i, j) = val+eps;
	objPlus = this->objective(svopn);
	this->verbMatrix[it->first].coeffRef(i, j) = val-eps;
	objMinus = this->objective(svopn);
	this->verbMatrix[it->first].coeffRef(i, j) = val;
	printf("backprop:  %.8f\n", this->verbGrad[it->first].coeff(i, j));
	printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
      }
    }
  }

  for (std::unordered_map<int, int>::iterator it = this->prepMap.begin(); it != this->prepMap.end(); ++it){
    printf("----------- prep %10d -------------\n", it->first);

    for (int i = 0; i < this->prepGrad[it->first].rows(); ++i){
      for (int j = 0; j < this->prepGrad[it->first].cols(); ++j){
	val = this->prepMatrix[it->first].coeff(i, j);
	this->prepMatrix[it->first].coeffRef(i, j) = val+eps;
	objPlus = this->objective(svopn);
	this->prepMatrix[it->first].coeffRef(i, j) = val-eps;
	objMinus = this->objective(svopn);
	this->prepMatrix[it->first].coeffRef(i, j) = val;
	printf("backprop:  %.8f\n", this->prepGrad[it->first].coeff(i, j));
	printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
      }
    }
  }
}

void NTF::save(const std::string& file){
  std::ofstream ofs(file.c_str(), std::ios::out|std::ios::binary);

  assert(ofs);
  Utils::save(ofs, this->nounVector);

  for (int i = 0; i < (int)this->verbMatrix.size(); ++i){
    Utils::save(ofs, this->verbMatrix[i]);
  }
  for (int i = 0; i < (int)this->prepMatrix.size(); ++i){
    Utils::save(ofs, this->prepMatrix[i]);
  }
}

void NTF::load(const std::string& file){
  std::ifstream ifs(file.c_str(), std::ios::out|std::ios::binary);

  assert(ifs);
  Utils::load(ifs, this->nounVector);

  for (int i = 0; i < (int)this->verbMatrix.size(); ++i){
    Utils::load(ifs, this->verbMatrix[i]);
  }
  for (int i = 0; i < (int)this->prepMatrix.size(); ++i){
    Utils::load(ifs, this->prepMatrix[i]);
  }
}
