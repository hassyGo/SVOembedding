#include "NTF.hpp"
#include "LBLM.hpp"

struct sort_pred {
  bool operator()(const std::pair<int, double> &left, const std::pair<int, double> &right) {
    return left.second > right.second;
  }
};

double calcRho(std::unordered_map<int, double>& rankMap, std::unordered_map<double, std::vector<int> >& tieMap, std::vector<std::pair<int, double> >& output){
  double rho = 0.0;

  std::sort(output.begin(), output.end(), sort_pred());
  
  for (int i = 0; i < (int)output.size(); ++i){
    double trueRank = 0.0;
    
    for (int j = 0; j < (int)tieMap.at(rankMap.at(output[i].first)).size(); ++j){
      trueRank += tieMap.at(rankMap.at(output[i].first))[j];
    }
    
    trueRank /= tieMap.at(rankMap.at(output[i].first)).size();
    rho += (trueRank-i)*(trueRank-i);
  }
  
  rho *= 6.0;
  rho /= (output.size()*output.size()*output.size()-output.size());
  rho = 1.0-rho;
  
  return rho;
}

double NTF::testSVO(Vocabulary& voc, const std::string& type, const bool ave){
  std::string file;

  if (type == "grefen"){
    file = (ave ? "./test_data/SVO_ave.txt" : "./test_data/SVO.txt");
  }
  else if (type == "emnlp2013"){
    file = (ave ? "./test_data/VO+S_ave.txt" : "./test_data/VO+S.txt");
  }
  else if (type == "emnlp2013add"){
    file = (ave ? "./test_data/VO+Sadd_ave.txt" : "./test_data/VO+Sadd.txt");
  }
  else {
    assert(false);
  }

  std::ifstream ifs(file.c_str());
  std::string line;
  int rank = 0;
  std::unordered_map<int, double> rankMap;
  std::unordered_map<double, std::vector<int> > tieMap;
  std::vector<std::pair<int, double> > output;
  
  while(std::getline(ifs, line) && line != ""){
    std::vector<std::string> res;
    double goldScore;
    MatD s1, s2;
    
    Utils::split(line, res);

    if (type == "grefen"){
      int s = voc.nounIndex.at(res[1]);
      int v = voc.verbIndex.at(res[0]);
      int o = voc.nounIndex.at(res[2]);
      int v_ = voc.verbIndex.at(res[3]);

      goldScore = atof(res[4].c_str());
      
      s1 = this->nounVector.col(s).array()*(this->verbMatrix[v]*this->nounVector.col(o)).array();
      s2 = this->nounVector.col(s).array()*(this->verbMatrix[v_]*this->nounVector.col(o)).array();
    }
    else {
      int s = voc.nounIndex.at(res[0]);
      int v = voc.verbIndex.at(res[1]);
      int o = voc.nounIndex.at(res[2]);
      int s_ = voc.nounIndex.at(res[3]);
      int v_ = voc.verbIndex.at(res[4]);
      int o_ = voc.nounIndex.at(res[5]);
     
      goldScore = atof(res[6].c_str());
      
      s1 = this->nounVector.col(s).array()*(this->verbMatrix[v]*this->nounVector.col(o)).array();
      s2 = this->nounVector.col(s_).array()*(this->verbMatrix[v_]*this->nounVector.col(o_)).array();
    }

    rankMap[rank] = goldScore;
    tieMap[goldScore].push_back(rank);
    output.push_back(std::pair<int, double>(rank++, Utils::cosDis(s1, s2)));
  }
  
  return calcRho(rankMap, tieMap, output);
}

double NTF::testVO(Vocabulary& voc, const bool ave){
  std::string file = (ave ? "./test_data/VO_ave.txt" : "./test_data/VO.txt");
  std::ifstream ifs(file.c_str());
  std::string line;
  int rank = 0;
  std::unordered_map<int, double> rankMap;
  std::unordered_map<double, std::vector<int> > tieMap;
  std::vector<std::pair<int, double> > output;
  
  while(std::getline(ifs, line) && line != ""){
    std::vector<std::string> res;
    double goldScore;
    
    Utils::split(line, res);

    int v1 = voc.verbIndex.at(res[1]);
    int o1 = voc.nounIndex.at(res[0]);
    int v2 = voc.verbIndex.at(res[3]);
    int o2 = voc.nounIndex.at(res[2]);

    goldScore = atof(res[4].c_str());
    
    rankMap[rank] = goldScore;
    tieMap[goldScore].push_back(rank);
    
    MatD s1 = this->verbMatrix[v1]*this->nounVector.col(o1);
    MatD s2 = this->verbMatrix[v2]*this->nounVector.col(o2);
    
    output.push_back(std::pair<int, double>(rank++, Utils::cosDis(s1, s2)));
  }

  return calcRho(rankMap, tieMap, output);
}

double LBLM::testSVO(Vocabulary& voc, const std::string& type, const bool ave){
  std::string file;

  if (type == "grefen"){
    file = (ave ? "./test_data/SVO_ave.txt" : "./test_data/SVO.txt");
  }
  else if (type == "emnlp2013"){
    file = (ave ? "./test_data/VO+S_ave.txt" : "./test_data/VO+S.txt");
  }
  else if (type == "emnlp2013add"){
    file = (ave ? "./test_data/VO+Sadd_ave.txt" : "./test_data/VO+Sadd.txt");
  }
  else {
    assert(false);
  }

  std::ifstream ifs(file.c_str());
  std::string line;
  int rank = 0;
  std::unordered_map<int, double> rankMap;
  std::unordered_map<double, std::vector<int> > tieMap;
  std::vector<std::pair<int, double> > output;
  
  while(std::getline(ifs, line) && line != ""){
    std::vector<std::string> res;
    double goldScore;
    MatD s1, s2;
    SVO svo;
    
    Utils::split(line, res);

    if (type == "grefen"){
      int s = voc.nounIndex.at(res[1]);
      int v = voc.verbIndex.at(res[0]);
      int o = voc.nounIndex.at(res[2]);
      int v_ = voc.verbIndex.at(res[3]);
      
      goldScore = atof(res[4].c_str());
      
      svo.s = s; svo.v = v; svo.o = o;
      this->compose(s1, &svo);
      svo.v = v_;
      this->compose(s2, &svo);
    }
    else {
      int s = voc.nounIndex.at(res[0]);
      int v = voc.verbIndex.at(res[1]);
      int o = voc.nounIndex.at(res[2]);
      int s_ = voc.nounIndex.at(res[3]);
      int v_ = voc.verbIndex.at(res[4]);
      int o_ = voc.nounIndex.at(res[5]);
      
      goldScore = atof(res[6].c_str());
      
      svo.s = s; svo.v = v; svo.o = o;
      this->compose(s1, &svo);
      svo.s = s_; svo.v = v_; svo.o = o_;
      this->compose(s2, &svo);
    }

    rankMap[rank] = goldScore;
    tieMap[goldScore].push_back(rank);
    output.push_back(std::pair<int, double>(rank++, Utils::cosDis(s1, s2)));
  }
  
  return calcRho(rankMap, tieMap, output);
}

double LBLM::testVO(Vocabulary& voc, const bool ave){
  std::string file = (ave ? "./test_data/VO_ave.txt" : "./test_data/VO.txt");
  std::ifstream ifs(file.c_str());
  std::string line;
  int rank = 0;
  std::unordered_map<int, double> rankMap;
  std::unordered_map<double, std::vector<int> > tieMap;
  std::vector<std::pair<int, double> > output;
  
  while(std::getline(ifs, line) && line != ""){
    std::vector<std::string> res;
    double goldScore;
    
    Utils::split(line, res);

    int v1 = voc.verbIndex.at(res[1]);
    int o1 = voc.nounIndex.at(res[0]);
    int v2 = voc.verbIndex.at(res[3]);
    int o2 = voc.nounIndex.at(res[2]);

    goldScore = atof(res[4].c_str());
    
    rankMap[rank] = goldScore;
    tieMap[goldScore].push_back(rank);
    
    SVO svo;
    MatD s1, s2;

    svo.s = voc.nullIndex; svo.v = v1; svo.o = o1;
    this->compose(s1, &svo);
    svo.s = voc.nullIndex; svo.v = v2; svo.o = o2;
    this->compose(s2, &svo);
    
    output.push_back(std::pair<int, double>(rank++, Utils::cosDis(s1, s2)));
  }

  return calcRho(rankMap, tieMap, output);
}
