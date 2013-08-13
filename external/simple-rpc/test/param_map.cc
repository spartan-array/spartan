#include "param_map.h"

#include <iostream>

namespace base {

using std::cout;
using std::endl;
using std::make_pair;
using std::pair;

ParamMap::ParamMap() {
  addParam("help", "", "prints the command usage message");
}

bool ParamMap::addParam(const string& param,
                        const string& default_string,
                        const string& description) {
  pair<KeyToValueMap::iterator, bool> res;
  res = key_to_value_map_.insert(make_pair(param, default_string));
  if (res.second) {
    res = key_to_description_map_.insert(make_pair(param, description));
    return true;
  }

  return false;
}

bool ParamMap::getParam(const string& param, string* value) const {
  // The 'value' parameter is optional.
  string dummy;
  if (value == NULL) {
    value = &dummy;
  }

  KeyToValueMap::const_iterator it = key_to_value_map_.find(param);
  if (it != key_to_value_map_.end()) {
    *value = it->second;
    return true;
  }

  return false;
}

bool ParamMap::parseArgv(int argc, const char* const argv[]) {
  program_name_ = argv[0];
  for (int i=1; i<argc; i++) {
    string name;
    string value;
    string param(argv[i]);

    // Param must start with "--".
    if (param.find("--") != 0) {
      return false;
    }

    // Param must have a name.
    string::size_type pos = param.find("=");
    if (pos == 2) {
      return false;
    }

    // Value for a parameter is not mandatory.
    if (pos == string::npos) {
      name = param.substr(2);
    } else {
      name = param.substr(2, pos-2);
      value = param.substr(pos+1);
    }

    if (name == "help") {
      return false;
    } else {
      key_to_value_map_[name] = value;
    }
  }

  return true;
}

void ParamMap::printUsage() const {
  cout << "Usage: " << program_name_ << " ";
  for (KeyToValueMap::const_iterator itp = key_to_value_map_.begin();
       itp != key_to_value_map_.end();
       ++itp) {
    cout << "--" << itp->first << "=[" << itp->second << "] ";
  }
  cout << endl;

  cout << "where " << endl;
  KeyToDescriptionMap::const_iterator itd;
  for (itd = key_to_description_map_.begin();
       itd != key_to_description_map_.end();
       ++itd) {
    cout << "  " << itd->first << ": " << itd->second << endl;
  }
}

} // namespace base
