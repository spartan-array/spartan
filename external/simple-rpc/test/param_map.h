#ifndef PARAM_MAP_HEADER
#define PARAM_MAP_HEADER

#include <string>
#include <unordered_map>

namespace base {

using std::string;
using std::unordered_map;

// Command-line argument parsing class. Arguments are declared first,
// and then parsed from 'argc' and 'argv'. The result is made
// available through a hash map from parameter name to parameter
// value.
//
// The class also produces "usage" descriptions.
class ParamMap {
public:
  ParamMap();
  ~ParamMap() {}

  // Returns true and inserts the pair 'param'->'default_string' in
  // the parameter map and 'param'->'description' in the description
  // map, if the pair doesn't exist alread. Return false otherwise.
  bool addParam(const string& param,
                const string& default_string,
                const string& description);

  // Returns true and optionally fills 'value', if one is provided,
  // with the currently value of'param', if it exists. Otherwise
  // return false.
  bool getParam(const string& param, string* value) const;

  // Returns true and parses each 'argv' into the format
  // --<name>[=<value>]. Insert each pair name, value into the
  // parameter map. Returns false if '--help' is one of the parameters
  // or if there are format issues.
  bool parseArgv(int argc, const char* const argv[]);

  // Output an usage string based on the parameters descriptions and
  // defaults of 'addParams' calls.
  void printUsage() const;

private:
  typedef unordered_map<string,string> KeyToValueMap;
  typedef unordered_map<string,string> KeyToDescriptionMap;

  KeyToValueMap       key_to_value_map_;
  KeyToDescriptionMap key_to_description_map_;
  string              program_name_;

  // Non-copyable, non-assignable
  ParamMap(const ParamMap&);
  ParamMap operator=(const ParamMap&);
};

} // namespace base

#endif // PARAM_MAP_HEADER
