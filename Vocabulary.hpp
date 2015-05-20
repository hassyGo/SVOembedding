#pragma once

#include "Matrix.hpp"
#include "Data.hpp"
#include "SVO.hpp"
#include "SVOPN.hpp"
#include <unordered_map>
#include <vector>

class Vocabulary{
public:
  static const std::string null;

  //verb
  std::unordered_map<std::string, int> verbIndex;
  std::vector<std::string> verbStr;
  std::vector<int> verbList;

  //preposition
  std::unordered_map<std::string, int> prepIndex;
  std::vector<std::string> prepStr;
  std::vector<int> prepList;

  //noun
  std::unordered_map<std::string, int> nounIndex;
  std::vector<std::string> nounStr;
  std::vector<int> vsubjList;
  std::vector<int> vobjList;
  std::vector<int> pobjList;
  int nullIndex;

  //svo
  std::vector<SVO*> svoList, svoListUniq;

  std::vector<SVOPN*> svopnListUniq;

  //svo map
  std::vector<std::unordered_map<int, std::unordered_map<int, SVO*> > > svoMap; //v -> s -> o
  std::vector<std::unordered_map<int, std::unordered_map<SVO*, SVOPN*> > > svopnMap; //p -> n -> svo

  Vocabulary(): nullIndex(-1) {};

  void readSVO(const std::string& fileName, std::vector<Data*>& instance, std::vector<Data*>& type);
  void readSVOPN(const std::string& fileName, std::vector<Data*>& instance, std::vector<Data*>& type);

  bool exist(int s, int v, int o);
  bool exist(int p, int n, SVO* svo);

private:
  static void split(const std::string& line, int& freq, std::string& pred, std::string& arg1, std::string& arg2);
};

inline bool Vocabulary::exist(int s, int v, int o){
  return this->svoMap[v].count(s) && this->svoMap[v].at(s).count(o);
}

inline bool Vocabulary::exist(int p, int n, SVO* svo){
  return this->svopnMap[p].count(n) && this->svopnMap[p].at(n).count(svo);
}
