#include "sparrow/util/file.h"
#include "sparrow/util/common.h"
#include "google/protobuf/message.h"
#include <stdio.h>
#include <glob.h>

using std::min;
using std::string;
using std::vector;

namespace sparrow {

static const int kFileBufferSize = 4 * 1024 * 1024;

vector<string> File::MatchingFilenames(StringPiece pattern) {
  glob_t globbuf;
  globbuf.gl_offs = 0;
  glob(pattern.AsString().c_str(), 0, NULL, &globbuf);
  vector<string> out;
  for (int i = 0; i < globbuf.gl_pathc; ++i) {
    out.push_back(globbuf.gl_pathv[i]);
  }
  globfree(&globbuf);
  return out;
}

vector<File::Info> File::MatchingFileinfo(StringPiece glob) {
  vector<string> names = MatchingFilenames(glob);
  vector<File::Info> out(names.size());
  for (int i = 0; i < names.size(); ++i) {
    out[i].name = names[i];
    stat(names[i].c_str(), &out[i].stat);
  }

  return out;
}

void File::Mkdirs(string path) {
  if (path[0] != '/') {
    char cur[PATH_MAX];
    getcwd(cur, PATH_MAX);
    path = string(cur) + "/" + path;
  }

  vector<StringPiece> pbits = StringPiece::split(path, "/");
  string prefix;
  for (int i = 0; i < pbits.size(); ++i) {
    pbits[i].strip();
    if (pbits[i].size() == 0) {
      continue;
    }

    prefix += "/" + pbits[i].AsString();
    int result = mkdir(prefix.c_str(), 0777);
    PCHECK(result == 0 || errno == EEXIST) << "Failed to create directory "
                                              << path;
  }
}

string File::Slurp(const string& f) {
  FILE* fp = fopen(f.c_str(), "r");
  CHECK(fp != NULL) << "Failed to read input file " << f;

  string out;
  char buffer[32768];

  while (!feof(fp) && !ferror(fp)) {
    int read = fread(buffer, 1, 32768, fp);
    if (read > 0) {
      out.append(buffer, read);
    } else {
      break;
    }
  }

  return out;
}

bool File::Exists(const string& f) {
  FILE* fp = fopen(f.c_str(), "r");
  if (fp) {
    fclose(fp);
    return true;
  }
  return false;
}

void File::Dump(const string& f, StringPiece data) {
  FILE* fp = fopen(f.c_str(), "w+");
  if (!fp) {
    LOG(FATAL)<< "Failed to open output file " << f.c_str();}
  fwrite(data.data, 1, data.len, fp);
  fflush(fp);
  fclose(fp);
}

void File::Move(const string& src, const string&dst) {
  PCHECK(rename(src.c_str(), dst.c_str()) == 0);
}

bool LocalFile::read_line(string *out) {
  out->clear();
  out->resize(8192);
  char* res = fgets(&(*out)[0], out->size(), fp);
  out->resize(strlen(out->data()));
  return res != NULL;
}

int LocalFile::read(char *buffer, int len) {
  return fread(buffer, 1, len, fp);
}

int LocalFile::write(const char *buffer, int len) {
  return fwrite(buffer, 1, len, fp);
}

void LocalFile::Printf(const char* p, ...) {
  va_list args;
  va_start(args, p);
  write_string(VStringPrintf(p, args));
  va_end(args);
}

bool LocalFile::eof() {
  return feof(fp);
}

LocalFile::LocalFile(FILE* stream) {
  CHECK(stream != NULL);
  fp = stream;
  path = "<EXTERNAL FILE>";
  close_on_delete = false;
}

LocalFile::LocalFile(const string &name, const string& mode) {
  fp = fopen(name.c_str(), mode.c_str());
  PCHECK(fp != NULL) << "; failed to open file " << name << " with mode "
                        << mode;
  path = name;
  close_on_delete = true;
  setvbuf(fp, NULL, _IOFBF, kFileBufferSize);
}

}
