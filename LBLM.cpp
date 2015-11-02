#include "LBLM.hpp"
#include <iostream>

#define ETA 1.0

void LBLM::init(const int dim, Vocabulary& voc){
  const double scale1 = 1.0/dim;
  const double scale2 = 1.0/(dim*dim);

  this->nounVector = MatD(dim, voc.nounStr.size());
  Data::rndModel.gauss(this->nounVector, scale1);
  this->verbVector = MatD(dim, voc.verbStr.size());
  Data::rndModel.gauss(this->verbVector, scale1);
  this->prepVector = MatD(dim, voc.prepStr.size());
  Data::rndModel.gauss(this->prepVector, scale1);

  if (voc.nullIndex >= 0){
    this->nounVector.col(voc.nullIndex).fill(0.0);
  }

  this->subjCompWeight = MatD::Ones(dim, 1);
  this->verbCompWeight = MatD::Ones(dim, 1);
  this->objCompWeight = MatD::Ones(dim, 1);

  this->subjWeight = MatD(dim, 1);
  Data::rndModel.gauss(this->subjWeight, scale2);
  this->verbWeight = MatD(dim, 1);
  Data::rndModel.gauss(this->verbWeight, scale2);
  this->objWeight = MatD(dim, 1);
  Data::rndModel.gauss(this->objWeight, scale2);

  this->prepWeight = MatD(dim, 1);
  Data::rndModel.gauss(this->prepWeight, scale2);
  this->pobjWeight = MatD(dim, 1);
  Data::rndModel.gauss(this->pobjWeight, scale2);
  this->svoWeight = MatD(dim, 1);
  Data::rndModel.gauss(this->svoWeight, scale2);

  this->nounScoreWeight = MatD::Zero(this->nounVector.cols(), this->nounVector.rows());
  this->verbScoreWeight = MatD::Zero(this->verbVector.cols(), this->verbVector.rows());
  this->prepScoreWeight = MatD::Zero(this->prepVector.cols(), this->prepVector.rows());

  //gradient
  this->nounGrad = MatD::Zero(this->nounVector.rows(), this->nounVector.cols());
  this->verbGrad = MatD::Zero(this->verbVector.rows(), this->verbVector.cols());
  this->prepGrad = MatD::Zero(this->prepVector.rows(), this->prepVector.cols());

  this->scwGrad = MatD::Zero(this->subjCompWeight.rows(), this->subjCompWeight.cols());
  this->vcwGrad = MatD::Zero(this->verbCompWeight.rows(), this->verbCompWeight.cols());
  this->ocwGrad = MatD::Zero(this->objCompWeight.rows(), this->objCompWeight.cols());

  this->swGrad = MatD::Zero(this->subjWeight.rows(), this->subjWeight.cols());
  this->vwGrad = MatD::Zero(this->verbWeight.rows(), this->verbWeight.cols());
  this->owGrad = MatD::Zero(this->objWeight.rows(), this->objWeight.cols());

  this->pwGrad = MatD::Zero(this->prepWeight.rows(), this->prepWeight.cols());
  this->powGrad = MatD::Zero(this->pobjWeight.rows(), this->pobjWeight.cols());
  this->svowGrad = MatD::Zero(this->svoWeight.rows(), this->svoWeight.cols());

  this->nounScoreGrad = MatD::Zero(this->nounScoreWeight.rows(), this->nounScoreWeight.cols());
  this->verbScoreGrad = MatD::Zero(this->verbScoreWeight.rows(), this->verbScoreWeight.cols());
  this->prepScoreGrad = MatD::Zero(this->prepScoreWeight.rows(), this->prepScoreWeight.cols());

  //AdaGrad
  this->nounGradHist = MatD::Ones(this->nounVector.rows(), this->nounVector.cols())*ETA;
  this->verbGradHist = MatD::Ones(this->verbVector.rows(), this->verbVector.cols())*ETA;
  this->prepGradHist = MatD::Ones(this->prepVector.rows(), this->prepVector.cols())*ETA;

  this->scwGradHist = MatD::Ones(this->subjCompWeight.rows(), this->subjCompWeight.cols())*ETA;
  this->vcwGradHist = MatD::Ones(this->verbCompWeight.rows(), this->verbCompWeight.cols())*ETA;
  this->ocwGradHist = MatD::Ones(this->objCompWeight.rows(), this->objCompWeight.cols())*ETA;

  this->swGradHist = MatD::Ones(this->subjWeight.rows(), this->subjWeight.cols())*ETA;
  this->vwGradHist = MatD::Ones(this->verbWeight.rows(), this->verbWeight.cols())*ETA;
  this->owGradHist = MatD::Ones(this->objWeight.rows(), this->objWeight.cols())*ETA;

  this->pwGradHist = MatD::Ones(this->prepWeight.rows(), this->prepWeight.cols())*ETA;
  this->powGradHist = MatD::Ones(this->pobjWeight.rows(), this->pobjWeight.cols())*ETA;
  this->svowGradHist = MatD::Ones(this->svoWeight.rows(), this->svoWeight.cols())*ETA;

  this->nounScoreGradHist = MatD::Ones(this->nounScoreWeight.rows(), this->nounScoreWeight.cols())*ETA;
  this->verbScoreGradHist = MatD::Ones(this->verbScoreWeight.rows(), this->verbScoreWeight.cols())*ETA;
  this->prepScoreGradHist = MatD::Ones(this->prepScoreWeight.rows(), this->prepScoreWeight.cols())*ETA;
}

void LBLM::train(std::vector<Data*>& instance, std::vector<Data*>& type, std::vector<Data*>& dummy, Vocabulary& voc, const double learningRate, const int maxItr, const int miniBatchSize, const int numNeg){
  const int prog = instance.size()/10;
  int counter = 0;
  
  for (int itr = 0; itr < maxItr; ++itr){
    counter = 0;
    std::random_shuffle(instance.begin(), instance.end());
    ::printf("LBLM: itr %2d/%2d, learning rate %.6f, dim %d\n", itr+1, maxItr, learningRate, (int)this->nounVector.rows());

    for (int i = 0; i < (int)instance.size(); ++i){
      if (instance[i]->type == Data::SVO_){
	this->train((SVO*)instance[i], voc);
	//this->gradCheck((SVO::SVO*)instance[i]);
      }
      else {
	this->train((SVOPN*)instance[i], voc);
      }
      
      if ((i+1)%miniBatchSize == 0 || i == (int)instance.size()-1){
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

	if (sample->set == Data::DEV){
	  devCount += 1.0;
	}
	else {
	  continue;
	}

	double vo2s = this->score(sample->s, sample->v, sample->o, LBLM::S);
	double sv2o = this->score(sample->s, sample->v, sample->o, LBLM::O);
	double so2v = this->score(sample->s, sample->v, sample->o, LBLM::V);

	for (int k = 0; k < numNeg; ++k){
	  double vo2s_ = this->score(sample->s_[k], sample->v, sample->o, LBLM::S);
	  double sv2o_ = this->score(sample->s, sample->v, sample->o_[k], LBLM::O);
	  double so2v_ = this->score(sample->s, sample->v_[k], sample->o, LBLM::V);

	  if (vo2s > vo2s_ && sv2o > sv2o_ && so2v > so2v_){
	    accDev.coeffRef(k, 0) += 1.0;
	  }
	}
      }
      else if (type[j]->type == Data::SVOPN_){
	SVOPN* sample = (SVOPN*)type[j];

	if (sample->set == Data::DEV){
	  devCount += 1.0;
	}
	else {
	  continue;
	}

	double vo2s = this->score(sample->p, sample->n, sample->svo, LBLM::P);
	double sv2o = this->score(sample->p, sample->n, sample->svo, LBLM::N);

	for (int k = 0; k < numNeg; ++k){
	  double vo2s_ = this->score(sample->p_[k], sample->n, sample->svo, LBLM::P);
	  double sv2o_ = this->score(sample->p, sample->n_[k], sample->svo, LBLM::N);
	  
	  if (vo2s > vo2s_ && sv2o > sv2o_){
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

void LBLM::train(SVO* svo, Vocabulary& voc){
  static MatD predict;
  static MatD WVs, WVv, WVo;
  static double scorePos, scoreNeg;
  static MatD delta;

  WVs = this->subjWeight.array()*this->nounVector.col(svo->s).array();
  WVv = this->verbWeight.array()*this->verbVector.col(svo->v).array();
  WVo = this->objWeight.array()*this->nounVector.col(svo->o).array();
  
  //V
  do {
    this->negV = voc.verbList[(Data::rndData.next() >> 16)%voc.verbList.size()];
  } while (voc.exist(svo->s, this->negV, svo->o));

  predict = WVs+WVo;
  Tanh::tanh(predict);
  scorePos = (this->verbScoreWeight.row(svo->v)*predict).coeff(0, 0);
  scoreNeg = (this->verbScoreWeight.row(this->negV)*predict).coeff(0, 0);

  if (scorePos <= scoreNeg+1.0){
    this->verbScoreMap[this->negV] = 1;
    this->verbScoreMap[svo->v] = 1;
    this->verbScoreGrad.row(svo->v) -= predict.transpose();
    this->verbScoreGrad.row(this->negV) += predict.transpose();
    delta = Tanh::tanhPrime(predict).array()*(this->verbScoreWeight.row(this->negV)-this->verbScoreWeight.row(svo->v)).transpose().array();
    this->swGrad.array() += delta.array()*this->nounVector.col(svo->s).array();
    this->owGrad.array() += delta.array()*this->nounVector.col(svo->o).array();
    this->nounGrad.col(svo->s).array() += delta.array()*this->subjWeight.array();
    this->nounGrad.col(svo->o).array() += delta.array()*this->objWeight.array();
    this->nounMap[svo->s] = 1;
    this->nounMap[svo->o] = 1;
  }

  //S
  do {
    this->negS = voc.vsubjList[(Data::rndData.next() >> 16)%voc.vsubjList.size()];
  } while (voc.exist(this->negS, svo->v, svo->o));

  predict = WVv+WVo;
  Tanh::tanh(predict);
  scorePos = (this->nounScoreWeight.row(svo->s)*predict).coeff(0, 0);
  scoreNeg = (this->nounScoreWeight.row(this->negS)*predict).coeff(0, 0);

  if (scorePos <= scoreNeg+1.0){
    this->nounScoreMap[this->negS] = 1;
    this->nounScoreMap[svo->s] = 1;
    this->nounScoreGrad.row(svo->s) -= predict.transpose();
    this->nounScoreGrad.row(this->negS) += predict.transpose();
    delta = Tanh::tanhPrime(predict).array()*(this->nounScoreWeight.row(this->negS)-this->nounScoreWeight.row(svo->s)).transpose().array();
    this->vwGrad.array() += delta.array()*this->verbVector.col(svo->v).array();
    this->owGrad.array() += delta.array()*this->nounVector.col(svo->o).array();
    this->verbGrad.col(svo->v).array() += delta.array()*this->verbWeight.array();
    this->nounGrad.col(svo->o).array() += delta.array()*this->objWeight.array();
    this->verbMap[svo->v] = 1;
    this->nounMap[svo->o] = 1;
  }

  //O
  do {
    this->negO = voc.vobjList[(Data::rndData.next() >> 16)%voc.vobjList.size()];
  } while (voc.exist(svo->s, svo->v, this->negO) || voc.exist(this->negS, svo->v, this->negO));

  predict = WVs+WVv;
  Tanh::tanh(predict);
  scorePos = (this->nounScoreWeight.row(svo->o)*predict).coeff(0, 0);
  scoreNeg = (this->nounScoreWeight.row(this->negO)*predict).coeff(0, 0);

  if (scorePos <= scoreNeg+1.0){
    this->nounScoreMap[this->negO] = 1;
    this->nounScoreMap[svo->o] = 1;
    this->nounScoreGrad.row(svo->o) -= predict.transpose();
    this->nounScoreGrad.row(this->negO) += predict.transpose();
    delta = Tanh::tanhPrime(predict).array()*(this->nounScoreWeight.row(this->negO)-this->nounScoreWeight.row(svo->o)).transpose().array();
    this->vwGrad.array() += delta.array()*this->verbVector.col(svo->v).array();
    this->swGrad.array() += delta.array()*this->nounVector.col(svo->s).array();
    this->verbGrad.col(svo->v).array() += delta.array()*this->verbWeight.array();
    this->nounGrad.col(svo->s).array() += delta.array()*this->subjWeight.array();
    this->verbMap[svo->v] = 1;
    this->nounMap[svo->s] = 1;
  }
}

void LBLM::train(SVOPN* svopn, Vocabulary& voc){
  static MatD predict;
  static MatD svo, svoPrime;
  static MatD WVp, WVn, WVsvo;
  static double scorePos, scoreNeg;
  static MatD delta, deltaComp = MatD(this->nounVector.rows(), 1);
  static bool flg;

  this->compose(svo, svopn->svo);
  svoPrime = Tanh::tanhPrime(svo);
  svoPrime.array() *= this->svoWeight.array();
  WVp = this->prepWeight.array()*this->prepVector.col(svopn->p).array();
  WVn = this->pobjWeight.array()*this->nounVector.col(svopn->n).array();
  WVsvo = this->svoWeight.array()*svo.array();
  
  deltaComp.setZero();
  flg = false;

  //P
  do {
    this->negP = voc.prepList[(Data::rndData.next() >> 16)%voc.prepList.size()];
  } while (voc.exist(this->negP, svopn->n, svopn->svo));

  predict = WVn+WVsvo;
  Tanh::tanh(predict);
  scorePos = (this->prepScoreWeight.row(svopn->p)*predict).coeff(0, 0);
  scoreNeg = (this->prepScoreWeight.row(this->negP)*predict).coeff(0, 0);

  if (scorePos <= scoreNeg+1.0){
    this->prepScoreGrad.row(svopn->p) -= predict.transpose();
    this->prepScoreGrad.row(this->negP) += predict.transpose();
    this->prepScoreMap[this->negP] = 1;
    this->prepScoreMap[svopn->p] = 1;

    delta = Tanh::tanhPrime(predict).array()*(this->prepScoreWeight.row(this->negP)-this->prepScoreWeight.row(svopn->p)).transpose().array();

    this->powGrad.array() += delta.array()*this->nounVector.col(svopn->n).array();
    this->svowGrad.array() += delta.array()*svo.array();

    this->nounGrad.col(svopn->n).array() += delta.array()*this->pobjWeight.array();
    this->nounMap[svopn->n] = 1;

    deltaComp.array() += delta.array()*svoPrime.array();
    flg = true;
  }

  //N
  do {
    this->negN = voc.pobjList[(Data::rndData.next() >> 16)%voc.pobjList.size()];
  } while (voc.exist(svopn->p, this->negN, svopn->svo));

  predict = WVp+WVsvo;
  Tanh::tanh(predict);
  scorePos = (this->nounScoreWeight.row(svopn->n)*predict).coeff(0, 0);
  scoreNeg = (this->nounScoreWeight.row(this->negN)*predict).coeff(0, 0);

  if (scorePos <= scoreNeg+1.0){
    this->nounScoreGrad.row(svopn->n) -= predict.transpose();
    this->nounScoreGrad.row(this->negN) += predict.transpose();
    this->nounScoreMap[this->negN] = 1;
    this->nounScoreMap[svopn->n] = 1;

    delta = Tanh::tanhPrime(predict).array()*(this->nounScoreWeight.row(this->negN)-this->nounScoreWeight.row(svopn->n)).transpose().array();

    this->pwGrad.array() += delta.array()*this->prepVector.col(svopn->p).array();
    this->svowGrad.array() += delta.array()*svo.array();

    this->prepGrad.col(svopn->p).array() += delta.array()*this->prepWeight.array();
    this->prepMap[svopn->p] = 1;

    deltaComp.array() += delta.array()*svoPrime.array();
    flg = true;
  }

  if (flg){
    this->nounGrad.col(svopn->svo->s).array() += deltaComp.array()*this->subjCompWeight.array();
    this->verbGrad.col(svopn->svo->v).array() += deltaComp.array()*this->verbCompWeight.array();
    this->nounGrad.col(svopn->svo->o).array() += deltaComp.array()*this->objCompWeight.array();
    this->scwGrad.array() += deltaComp.array()*this->nounVector.col(svopn->svo->s).array();
    this->vcwGrad.array() += deltaComp.array()*this->verbVector.col(svopn->svo->v).array();
    this->ocwGrad.array() += deltaComp.array()*this->nounVector.col(svopn->svo->o).array();
    this->nounMap[svopn->svo->s] = 1;
    this->verbMap[svopn->svo->v] = 1;
    this->nounMap[svopn->svo->o] = 1;
  }
}

void LBLM::update(const double learningRate, const int exception){
  for (auto it = this->nounMap.begin(); it != this->nounMap.end(); ++it){
    if (it->first == exception){
      this->nounGrad.col(it->first).setZero();
      continue;
    }

    this->nounGradHist.col(it->first).array() += this->nounGrad.col(it->first).array().square();
    this->nounGrad.col(it->first).array() /= this->nounGradHist.col(it->first).array().sqrt();
    this->nounVector.col(it->first) -= learningRate*this->nounGrad.col(it->first);
    this->nounGrad.col(it->first).setZero();
  }
  for (auto it = this->verbMap.begin(); it != this->verbMap.end(); ++it){
    this->verbGradHist.col(it->first).array() += this->verbGrad.col(it->first).array().square();
    this->verbGrad.col(it->first).array() /= this->verbGradHist.col(it->first).array().sqrt();
    this->verbVector.col(it->first) -= learningRate*this->verbGrad.col(it->first);
    this->verbGrad.col(it->first).setZero();
  }
  for (auto it = this->prepMap.begin(); it != this->prepMap.end(); ++it){
    this->prepGradHist.col(it->first).array() += this->prepGrad.col(it->first).array().square();
    this->prepGrad.col(it->first).array() /= this->prepGradHist.col(it->first).array().sqrt();
    this->prepVector.col(it->first) -= learningRate*this->prepGrad.col(it->first);
    this->prepGrad.col(it->first).setZero();
  }

  for (auto it = this->nounScoreMap.begin(); it != this->nounScoreMap.end(); ++it){
    this->nounScoreGradHist.row(it->first).array() += this->nounScoreGrad.row(it->first).array().square();
    this->nounScoreGrad.row(it->first).array() /= this->nounScoreGradHist.row(it->first).array().sqrt();
    this->nounScoreWeight.row(it->first) -= learningRate*this->nounScoreGrad.row(it->first);
    this->nounScoreGrad.row(it->first).setZero();
  }
  for (auto it = this->verbScoreMap.begin(); it != this->verbScoreMap.end(); ++it){
    this->verbScoreGradHist.row(it->first).array() += this->verbScoreGrad.row(it->first).array().square();
    this->verbScoreGrad.row(it->first).array() /= this->verbScoreGradHist.row(it->first).array().sqrt();
    this->verbScoreWeight.row(it->first) -= learningRate*this->verbScoreGrad.row(it->first);
    this->verbScoreGrad.row(it->first).setZero();
  }
  for (auto it = this->prepScoreMap.begin(); it != this->prepScoreMap.end(); ++it){
    this->prepScoreGradHist.row(it->first).array() += this->prepScoreGrad.row(it->first).array().square();
    this->prepScoreGrad.row(it->first).array() /= this->prepScoreGradHist.row(it->first).array().sqrt();
    this->prepScoreWeight.row(it->first) -= learningRate*this->prepScoreGrad.row(it->first);
    this->prepScoreGrad.row(it->first).setZero();
  }

  this->nounMap.clear();
  this->verbMap.clear();
  this->prepMap.clear();
  this->nounScoreMap.clear();
  this->verbScoreMap.clear();
  this->prepScoreMap.clear();

  this->scwGradHist.array() += this->scwGrad.array().square();
  this->scwGrad.array() /= this->scwGradHist.array().sqrt();
  this->subjCompWeight -= learningRate*this->scwGrad;

  this->vcwGradHist.array() += this->vcwGrad.array().square();
  this->vcwGrad.array() /= this->vcwGradHist.array().sqrt();
  this->verbCompWeight -= learningRate*this->vcwGrad;

  this->ocwGradHist.array() += this->ocwGrad.array().square();
  this->ocwGrad.array() /= this->ocwGradHist.array().sqrt();
  this->objCompWeight -= learningRate*this->ocwGrad;

  this->swGradHist.array() += this->swGrad.array().square();
  this->swGrad.array() /= this->swGradHist.array().sqrt();
  this->subjWeight -= learningRate*this->swGrad;

  this->vwGradHist.array() += this->vwGrad.array().square();
  this->vwGrad.array() /= this->vwGradHist.array().sqrt();
  this->verbWeight -= learningRate*this->vwGrad;

  this->owGradHist.array() += this->owGrad.array().square();
  this->owGrad.array() /= this->owGradHist.array().sqrt();
  this->objWeight -= learningRate*this->owGrad;

  this->pwGradHist.array() += this->pwGrad.array().square();
  this->pwGrad.array() /= this->pwGradHist.array().sqrt();
  this->prepWeight -= learningRate*this->pwGrad;

  this->powGradHist.array() += this->powGrad.array().square();
  this->powGrad.array() /= this->powGradHist.array().sqrt();
  this->pobjWeight -= learningRate*this->powGrad;

  this->svowGradHist.array() += this->svowGrad.array().square();
  this->svowGrad.array() /= this->svowGradHist.array().sqrt();
  this->svoWeight -= learningRate*this->svowGrad;

  this->scwGrad.setZero();
  this->vcwGrad.setZero();
  this->ocwGrad.setZero();
  this->swGrad.setZero();
  this->vwGrad.setZero();
  this->owGrad.setZero();
  this->pwGrad.setZero();
  this->powGrad.setZero();
  this->svowGrad.setZero();
}

void LBLM::gradCheck(SVO* svo){
  const double eps = 1.0e-04;
  double val, objPlus, objMinus;

  printf("\nchecking gradients ...\n");

  for (auto it = this->nounMap.begin(); it != this->nounMap.end(); ++it){
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

  for (auto it = this->verbMap.begin(); it != this->verbMap.end(); ++it){
    printf("----------- verb %10d -------------\n", it->first);
    for (int i = 0; i < this->verbGrad.rows(); ++i){
      val = this->verbVector.coeff(i, it->first);
      this->verbVector.coeffRef(i, it->first) = val+eps;
      objPlus = this->objective(svo);
      this->verbVector.coeffRef(i, it->first) = val-eps;
      objMinus = this->objective(svo);
      this->verbVector.coeffRef(i, it->first) = val;
      printf("backprop:  %.8f\n", this->verbGrad.coeff(i, it->first));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
    }
  }

  for (auto it = this->nounScoreMap.begin(); it != this->nounScoreMap.end(); ++it){
    printf("----------- noun score weight %10d -------------\n", it->first);
    for (int i = 0; i < this->nounScoreGrad.cols(); ++i){
      val = this->nounScoreWeight.coeff(it->first, i);
      this->nounScoreWeight.coeffRef(it->first, i) = val+eps;
      objPlus = this->objective(svo);
      this->nounScoreWeight.coeffRef(it->first, i) = val-eps;
      objMinus = this->objective(svo);
      this->nounScoreWeight.coeffRef(it->first, i) = val;
      printf("backprop:  %.8f\n", this->nounScoreGrad.coeff(it->first, i));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
    }
  }

  for (auto it = this->verbScoreMap.begin(); it != this->verbScoreMap.end(); ++it){
    printf("----------- verb score weight %10d -------------\n", it->first);
    for (int i = 0; i < this->verbScoreGrad.cols(); ++i){
      val = this->verbScoreWeight.coeff(it->first, i);
      this->verbScoreWeight.coeffRef(it->first, i) = val+eps;
      objPlus = this->objective(svo);
      this->verbScoreWeight.coeffRef(it->first, i) = val-eps;
      objMinus = this->objective(svo);
      this->verbScoreWeight.coeffRef(it->first, i) = val;
      printf("backprop:  %.8f\n", this->verbScoreGrad.coeff(it->first, i));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));
    }
  }

  printf("----------- subj weight -------------\n");
  for (int i = 0; i < this->subjWeight.rows(); ++i){
    for (int j = 0; j < this->subjWeight.cols(); ++j){
      val = this->subjWeight.coeff(i, j);
      this->subjWeight.coeffRef(i, j) = val+eps;
      objPlus = this->objective(svo);
      this->subjWeight.coeffRef(i, j) = val-eps;
      objMinus = this->objective(svo);
      this->subjWeight.coeffRef(i, j) = val;
      printf("backprop:  %.8f\n", this->swGrad.coeff(i, j));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));      
    }
  }

  printf("----------- verb weight -------------\n");
  for (int i = 0; i < this->verbWeight.rows(); ++i){
    for (int j = 0; j < this->verbWeight.cols(); ++j){
      val = this->verbWeight.coeff(i, j);
      this->verbWeight.coeffRef(i, j) = val+eps;
      objPlus = this->objective(svo);
      this->verbWeight.coeffRef(i, j) = val-eps;
      objMinus = this->objective(svo);
      this->verbWeight.coeffRef(i, j) = val;
      printf("backprop:  %.8f\n", this->vwGrad.coeff(i, j));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));      
    }
  }

  printf("----------- obj weight -------------\n");
  for (int i = 0; i < this->objWeight.rows(); ++i){
    for (int j = 0; j < this->objWeight.cols(); ++j){
      val = this->objWeight.coeff(i, j);
      this->objWeight.coeffRef(i, j) = val+eps;
      objPlus = this->objective(svo);
      this->objWeight.coeffRef(i, j) = val-eps;
      objMinus = this->objective(svo);
      this->objWeight.coeffRef(i, j) = val;
      printf("backprop:  %.8f\n", this->owGrad.coeff(i, j));
      printf("numerical: %.8f\n", (objPlus-objMinus)/(2.0*eps));      
    }
  }
}

void LBLM::save(const std::string& file){
  std::ofstream ofs(file.c_str(), std::ios::out|std::ios::binary);

  assert(ofs);
  Utils::save(ofs, this->nounVector);
  Utils::save(ofs, this->verbVector);
  Utils::save(ofs, this->prepVector);

  Utils::save(ofs, this->subjCompWeight);
  Utils::save(ofs, this->verbCompWeight);
  Utils::save(ofs, this->objCompWeight);

  Utils::save(ofs, this->subjWeight);
  Utils::save(ofs, this->verbWeight);
  Utils::save(ofs, this->objWeight);

  Utils::save(ofs, this->prepWeight);
  Utils::save(ofs, this->pobjWeight);
  Utils::save(ofs, this->svoWeight);

  Utils::save(ofs, this->nounScoreWeight);
  Utils::save(ofs, this->verbScoreWeight);
  Utils::save(ofs, this->prepScoreWeight);
}

void LBLM::load(const std::string& file){
  std::ifstream ifs(file.c_str(), std::ios::out|std::ios::binary);

  assert(ifs);
  Utils::load(ifs, this->nounVector);
  Utils::load(ifs, this->verbVector);
  Utils::load(ifs, this->prepVector);

  Utils::load(ifs, this->subjCompWeight);
  Utils::load(ifs, this->verbCompWeight);
  Utils::load(ifs, this->objCompWeight);

  Utils::load(ifs, this->subjWeight);
  Utils::load(ifs, this->verbWeight);
  Utils::load(ifs, this->objWeight);

  Utils::load(ifs, this->prepWeight);
  Utils::load(ifs, this->pobjWeight);
  Utils::load(ifs, this->svoWeight);

  Utils::load(ifs, this->nounScoreWeight);
  Utils::load(ifs, this->verbScoreWeight);
  Utils::load(ifs, this->prepScoreWeight);
}
