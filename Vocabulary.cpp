#include "Vocabulary.hpp"
#include "Utils.hpp"
#include <fstream>

const std::string Vocabulary::null = "NULL";

void Vocabulary::split(const std::string& line, int& freq, std::string& pred, std::string& arg1, std::string& arg2){
  for (int i = line.length()-1, space = 0, s1 = -1, s2 = -1, s3 = -1; ; --i){
    if (line[i] == ' '){
      if (space == 0){
	s1 = i;
	++space;
	arg2 = line.substr(s1+1);
      }
      else if (space == 1){
	s2 = i;
	++space;
	arg1 = line.substr(s2+1, s1-s2-1);
      }
      else if (space == 2){
	s3 = i;
	++space;
	pred = line.substr(s3+1, s2-s3-1);
	freq = atoi(line.substr(0, s3).c_str());
	break;
      }
    }
  }

  assert(pred.find(" ") == std::string::npos && arg1.find(" ") == std::string::npos && arg2.find(" ") == std::string::npos);
}

void Vocabulary::readSVO(const std::string& fileName, std::vector<Data*>& instance, std::vector<Data*>& type){
  std::ifstream ifs(fileName.c_str());
  std::string subj, obj, verb;
  SVO* svo = 0;
  std::unordered_map<std::string, int>::iterator it;
  int freq = -1;
  std::unordered_map<int, int> subjCount, objCount, verbCount;

  for (std::string line; std::getline(ifs, line);){
    Vocabulary::split(line, freq, verb, subj, obj);

    //only use SVO tuples
    if (verb == "be" || subj == Vocabulary::null || obj == Vocabulary::null){
      continue;
    }

    svo = new SVO;
    svo->type = Data::SVO_;

    //object
    it = this->nounIndex.find(obj);
    if (it == this->nounIndex.end()){
      this->nounStr.push_back(obj);
      this->nounIndex[obj] = this->nounStr.size()-1;
    }
    svo->o = this->nounIndex.at(obj);

    //subject
    it = this->nounIndex.find(subj);
    if (it == this->nounIndex.end()){
      this->nounStr.push_back(subj);
      this->nounIndex[subj] = this->nounStr.size()-1;
    }
    svo->s = this->nounIndex.at(subj);

    //verb
    it = this->verbIndex.find(verb);
    if (it == this->verbIndex.end()){
      this->verbStr.push_back(verb);
      this->verbIndex[verb] = this->verbStr.size()-1;
      this->svoMap.push_back(std::unordered_map<int, std::unordered_map<int, SVO*> >());
    }
    svo->v = this->verbIndex.at(verb);

    //add this svo
    if (!this->svoMap[svo->v].count(svo->s)){
      this->svoMap[svo->v][svo->s] = std::unordered_map<int, SVO*>();
    }
    this->svoMap[svo->v].at(svo->s)[svo->o] = svo;

    type.push_back(svo);

    for (int j = 0; j < freq; ++j){
      instance.push_back(svo);
    }

    if (subjCount.count(svo->s)){
      subjCount.at(svo->s) += freq;
    }
    else {
      subjCount[svo->s] = freq;
    }
    if (objCount.count(svo->o)){
      objCount.at(svo->o) += freq;
    }
    else {
      objCount[svo->o] = freq;
    }
    if (verbCount.count(svo->v)){
      verbCount.at(svo->v) += freq;
    }
    else {
      verbCount[svo->v] = freq;
    }
  }

  for (std::unordered_map<int, int>::iterator it = subjCount.begin(); it != subjCount.end(); ++it){
    for (int i = 0; i < (int)pow(it->second, 0.75); ++i){
      this->vsubjList.push_back(it->first);
    }
  }
  for (std::unordered_map<int, int>::iterator it = objCount.begin(); it != objCount.end(); ++it){
    for (int i = 0; i < (int)pow(it->second, 0.75); ++i){
      this->vobjList.push_back(it->first);
    }
  }
  for (std::unordered_map<int, int>::iterator it = verbCount.begin(); it != verbCount.end(); ++it){
    for (int i = 0; i < (int)pow(it->second, 0.75); ++i){
      this->verbList.push_back(it->first);
    }
  }

  printf("# of SVO types: %zd\n", type.size());
  printf("# of SVO instances: %zd\n", instance.size());
}

void Vocabulary::readSVOPN(const std::string& fileName, std::vector<Data*>& instance, std::vector<Data*>& type){
  std::ifstream ifs(fileName.c_str());
  std::string prep, svo, n;
  std::string subj, verb, obj;
  int c1 = -1, c2 = -1;
  int s = -1, v = -1, o = -1;
  SVOPN* svopn = 0;
  std::unordered_map<std::string, int>::iterator it;
  int freq = -1;
  std::unordered_map<int, int> prepCount, pobjCount;
  std::unordered_map<SVO*, int> svoIndex;
  std::vector<std::pair<SVO*, int> > svoCount;

  for (std::string line; std::getline(ifs, line);){
    Vocabulary::split(line, freq, prep, svo, n);

    //check SVO
    c1 = svo.find(",");
    c2 = svo.rfind(",");
    verb = svo.substr(0, c1);
    subj = svo.substr(c1+1, c2-c1-1);
    obj = svo.substr(c2+1);

    if (obj == Vocabulary::null || verb == "be"){
      continue;
    }

    svopn = new SVOPN;
    svopn->type = Data::SVOPN_;

    //n
    it = this->nounIndex.find(n);
    if (it == this->nounIndex.end()){
      this->nounStr.push_back(n);
      this->nounIndex[n] = this->nounStr.size()-1;
    }
    svopn->n = this->nounIndex.at(n);
    
    //svo
    it = this->nounIndex.find(obj);
    if (it == this->nounIndex.end()){
      this->nounStr.push_back(obj);
      this->nounIndex[obj] = this->nounStr.size()-1;
    }
    it = this->nounIndex.find(subj);
    if (it == this->nounIndex.end()){
      this->nounStr.push_back(subj);
      this->nounIndex[subj] = this->nounStr.size()-1;
    }
    it = this->verbIndex.find(verb);
    if (it == this->verbIndex.end()){
      this->verbStr.push_back(verb);
      this->verbIndex[verb] = this->verbStr.size()-1;
      this->svoMap.push_back(std::unordered_map<int, std::unordered_map<int, SVO*> >());
    }
    s = this->nounIndex.at(subj);
    o = this->nounIndex.at(obj);
    v = this->verbIndex.at(verb);
    if (this->exist(s, v, o)){
      svopn->svo = this->svoMap[v].at(s).at(o);
    }
    else {
      svopn->svo = new SVO;
      svopn->svo->type = Data::SVO_;
      svopn->svo->s = s;
      svopn->svo->v = v;
      svopn->svo->o = o;
      if (!this->svoMap[v].count(s)){
	this->svoMap[v][s] = std::unordered_map<int, SVO*>();
      }
      this->svoMap[v].at(s)[o] = svopn->svo;
    }

    //prep
    it = this->prepIndex.find(prep);
    if (it == this->prepIndex.end()){
      this->prepStr.push_back(prep);
      this->prepIndex[prep] = this->prepStr.size()-1;
      this->svopnMap.push_back(std::unordered_map<int, std::unordered_map<SVO*, SVOPN*> >());
    }
    svopn->p = this->prepIndex.at(prep);

    //add this svo-pn
    if (!this->svopnMap[svopn->p].count(svopn->n)){
      this->svopnMap[svopn->p][svopn->n] = std::unordered_map<SVO*, SVOPN*>();
    }

    this->svopnMap[svopn->p].at(svopn->n)[svopn->svo] = svopn;

    type.push_back(svopn);

    for (int j = 0; j < freq; ++j){
      instance.push_back(svopn);
    }

    if (prepCount.count(svopn->p)){
      prepCount.at(svopn->p) += freq;
    }
    else {
      prepCount[svopn->p] = freq;
    }
    if (pobjCount.count(svopn->n)){
      pobjCount.at(svopn->n) += freq;
    }
    else {
      pobjCount[svopn->n] = freq;
    }

    if (svoIndex.count(svopn->svo)){
      svoCount[svoIndex.at(svopn->svo)].second += freq;
    }
    else {
      svoIndex[svopn->svo] = svoCount.size();
      svoCount.push_back(std::pair<SVO*, int>(svopn->svo, freq));
    }
  }

  if (this->nounIndex.count(Vocabulary::null)){
    this->nullIndex = this->nounIndex.at(Vocabulary::null);
  }

  for (std::unordered_map<int, int>::iterator it = prepCount.begin(); it != prepCount.end(); ++it){
    for (int i = 0; i < (int)pow(it->second, 0.75); ++i){
      this->prepList.push_back(it->first);
    }
  }
  for (std::unordered_map<int, int>::iterator it = pobjCount.begin(); it != pobjCount.end(); ++it){
    for (int i = 0; i < (int)pow(it->second, 0.75); ++i){
      this->pobjList.push_back(it->first);
    }
  }
  for (int i = 0; i < (int)svoCount.size(); ++i){
    for (int j = 0; j < (int)pow(svoCount[i].second, 0.75); ++j){
      this->svoList.push_back(svoCount[i].first);
    }
  }

  printf("# of SVO-P-N types: %zd\n", type.size());
  printf("# of SVO-P-N instances: %zd\n", instance.size());
}
