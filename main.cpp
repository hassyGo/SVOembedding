#include "Vocabulary.hpp"
#include "Data.hpp"
#include "SVO.hpp"
#include "LBLM.hpp"
#include <iostream>

/*
  argv[1]: dim
  argv[2]: learning rate
  argv[3]: iteration
  argv[4]: svopn ? 1 or 0
  argv[5]: bnc or wiki
  argv[6]: save (or load) file
*/

int main(int argc, char** argv){
  const int dim = atoi(argv[1]);
  const double learningRate = atof(argv[2]);
  const int itr = atoi(argv[3]);
  const bool svopn = (atoi(argv[4]) == 1);
  const int miniBatchSize = 100;
  const std::string corpus = (std::string)argv[5];
  int numNeg = -1;
  Vocabulary voc;
  std::vector<Data*> instance, instanceSVO, instanceSVOPN;
  std::vector<Data*> typeSVO, typeSVOPN;
  std::vector<Data*> type;
  LBLM lblm;

  if (corpus == "bnc"){
    numNeg = 50;
    voc.readSVO("svo.bnc.count", instanceSVO, typeSVO);
    
    if (svopn){
      voc.readSVOPN("svo_pn.bnc.count", instanceSVOPN, typeSVOPN);
    }
  }
  else if (corpus == "wiki"){
    numNeg = 10;
    voc.readSVO("svo.wiki.count", instanceSVO, typeSVO);
    
    if (svopn){
      voc.readSVOPN("svo_pn.wiki.count", instanceSVOPN, typeSVOPN);
    }
  }
  else {
    assert(false);
  }

  for (int i = 0; i < (int)voc.svoMap.size(); ++i){
    for (std::unordered_map<int, std::unordered_map<int, SVO*> >::iterator it1 = voc.svoMap[i].begin(); it1 != voc.svoMap[i].end(); ++it1){
      for (std::unordered_map<int, SVO*>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2){
	voc.svoListUniq.push_back(it2->second);
      }
    }
  }
  for (int i = 0; i < (int)voc.svopnMap.size(); ++i){
    for (std::unordered_map<int, std::unordered_map<SVO*, SVOPN*> >::iterator it1 = voc.svopnMap[i].begin(); it1 != voc.svopnMap[i].end(); ++it1){
      for (std::unordered_map<SVO*, SVOPN*>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2){
	voc.svopnListUniq.push_back(it2->second);
      }
    }
  }
    
  //split
  for (int i = 0; i < (int)typeSVO.size(); ++i){
    double rndTmp = Data::rndData.zero2one();

    typeSVO[i]->set = (rndTmp <= 0.8 ?
		       Data::TRAIN :
		       (rndTmp <= 0.9 ? Data::DEV : Data::TEST));
    type.push_back(typeSVO[i]);

    if (typeSVO[i]->set == Data::TRAIN){
      continue;
    }
    
    ((SVO*)typeSVO[i])->s_ = new int[numNeg];
    ((SVO*)typeSVO[i])->v_ = new int[numNeg];
    ((SVO*)typeSVO[i])->o_ = new int[numNeg];
    
    for (int j = 0; j < numNeg; ++j){
      do {
	((SVO*)typeSVO[i])->s_[j] = voc.vsubjList[(Data::rndData.next() >> 16)%voc.vsubjList.size()];
      } while (voc.exist(((SVO*)typeSVO[i])->s_[j], ((SVO*)typeSVO[i])->v, ((SVO*)typeSVO[i])->o));
      do {
	((SVO*)typeSVO[i])->v_[j] = voc.verbList[(Data::rndData.next() >> 16)%voc.verbList.size()];
      } while (voc.exist(((SVO*)typeSVO[i])->s, ((SVO*)typeSVO[i])->v_[j], ((SVO*)typeSVO[i])->o));
      do {
	((SVO*)typeSVO[i])->o_[j] = voc.vobjList[(Data::rndData.next() >> 16)%voc.vobjList.size()];
      } while (voc.exist(((SVO*)typeSVO[i])->s, ((SVO*)typeSVO[i])->v, ((SVO*)typeSVO[i])->o_[j]));
    }
  }
  for (int i = 0; i < (int)typeSVOPN.size(); ++i){
    double rndTmp = Data::rndData.zero2one();

    typeSVOPN[i]->set = (rndTmp <= 0.8 ?
			 Data::TRAIN :
			 (rndTmp <= 0.9 ? Data::DEV : Data::TEST));
    type.push_back(typeSVOPN[i]);

    if (typeSVOPN[i]->set == Data::TRAIN){
      continue;
    }
    
    ((SVOPN*)typeSVOPN[i])->p_ = new int[numNeg];
    ((SVOPN*)typeSVOPN[i])->n_ = new int[numNeg];
    ((SVOPN*)typeSVOPN[i])->svo_ = new SVO*[numNeg];

    for (int j = 0; j < numNeg; ++j){
      do {
	((SVOPN*)typeSVOPN[i])->p_[j] = voc.prepList[(Data::rndData.next() >> 16)%voc.prepList.size()];
      } while (voc.exist(((SVOPN*)typeSVOPN[i])->p_[j], ((SVOPN*)typeSVOPN[i])->n, ((SVOPN*)typeSVOPN[i])->svo));
      do {
	((SVOPN*)typeSVOPN[i])->n_[j] = voc.pobjList[(Data::rndData.next() >> 16)%voc.pobjList.size()];
      } while (voc.exist(((SVOPN*)typeSVOPN[i])->p, ((SVOPN*)typeSVOPN[i])->n_[j], ((SVOPN*)typeSVOPN[i])->svo));
      do {
	((SVOPN*)typeSVOPN[i])->svo_[j] = voc.svoList[(Data::rndData.next() >> 16)%voc.svoList.size()];
      } while (voc.exist(((SVOPN*)typeSVOPN[i])->p, ((SVOPN*)typeSVOPN[i])->n, ((SVOPN*)typeSVOPN[i])->svo_[j]));
    }
  }
  
  for (int i = 0; i < (int)instanceSVO.size(); ++i){
    if (instanceSVO[i]->set == Data::TRAIN){
      instance.push_back(instanceSVO[i]);
    }
  }
  for (int i = 0; i < (int)instanceSVOPN.size(); ++i){
    if (instanceSVOPN[i]->set == Data::TRAIN){
      instance.push_back(instanceSVOPN[i]);
    }
  }
  
  lblm.init(dim, voc);
  lblm.train(instance, type, type, voc, learningRate, itr, miniBatchSize, numNeg);
  lblm.save((std::string)argv[6]);
  
  //lblm.load((std::string)argv[6]);
  lblm.VOdist(voc, 20);
  lblm.SVOdist(voc, 20);
  lblm.SVOdist(voc);

  return 0;
}
