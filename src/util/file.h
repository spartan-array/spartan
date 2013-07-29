#ifndef FILE_H_
#define FILE_H_

#include "sparrow.pb.h"

#include "boost/noncopyable.hpp"
#include "util/common.h"
#include "util/marshal.h"
#include "util/stringpiece.h"


#include <string>
#include <vector>

#include <glob.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace sparrow {

class File {
public:
  virtual ~File() {
  }
  virtual int read(char *buffer, int len) = 0;
  virtual bool read_line(std::string *out) = 0;
  virtual bool eof() = 0;
  virtual void seek(int64_t pos) = 0;
  virtual uint64_t tell() = 0;
  virtual const char* name() {
    return "";
  }
  virtual void sync() = 0;

  int write_string(const std::string& buffer) {
    return write(buffer.data(), buffer.size());
  }

  virtual int write(const char* buffer, int len) = 0;

  std::string readLine() {
    std::string out;
    read_line(&out);
    return out;
  }

  struct Info {
    std::string name;
    struct stat stat;
  };

  static std::string Slurp(const std::string& file);
  static void Dump(const std::string& file, StringPiece data);
  static void Mkdirs(std::string path);
  static std::vector<std::string> MatchingFilenames(StringPiece glob);
  static std::vector<Info> MatchingFileinfo(StringPiece glob);

  static bool Exists(const std::string& path);
  static void Move(const std::string& src, const std::string&dst);
private:
};

class LocalFile: public File {
public:
  LocalFile(FILE* fp);
  LocalFile(const std::string& path, const std::string& mode);
  virtual ~LocalFile() {
    if (close_on_delete) {
      fflush(fp);
      fclose(fp);
    }
  }

  void sync() {
    fsync(fileno(fp));
  }

  bool read_line(std::string *out);
  int read(char *buffer, int len);
  int write(const char* buffer, int len);
  void seek(int64_t pos) {
    fseek(fp, pos, SEEK_SET);
  }
  uint64_t tell() {
    return ftell(fp);
  }

  void Printf(const char* p, ...);
  virtual FILE* filePointer() {
    return fp;
  }

  const char* name() {
    return path.c_str();
  }

  bool eof();

private:
  FILE* fp;
  std::string path;
  bool close_on_delete;
};

class FileWriter: public Writer {
private:
  File* f_;

public:
  FileWriter(const string& filename) {
    f_ = new LocalFile(filename, "w");
  }

  ~FileWriter() {
    delete f_;
  }

  void write_bytes(const void* v, int sz) {
    f_->write((char*) v, sz);
  }
};

class FileReader: public Reader {
private:
  File* f_;
public:
  FileReader(const string& filename) {
    f_ = new LocalFile(filename, "r");
  }

  ~FileReader() {
    delete f_;
  }

  int read_bytes(void* v, int sz) {
    return f_->read((char*) v, sz);
  }
};
}

#endif /* FILE_H_ */
