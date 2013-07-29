// -*- c-file-style: "bsd"; c-file-offsets: ((innamespace . 0)) -*-

#include "webgraph.h"

#include <algorithm>
#include <iostream>
#include <map>

#include <assert.h>
#include <errno.h>
#include <zlib.h>

namespace WebGraph {

/******************************************************************
 * Byte reader
 */

class GZByteReader {
private:
  gzFile f;
  void raise();

public:
  GZByteReader(const char *path);
  ~GZByteReader();
  int read(void *buf, int size);
  int getc();
};

GZByteReader::GZByteReader(const char *path) {
  errno = 0;
  f = gzopen(path, "rb");
  if (f == NULL) {
    if (errno == ENOENT)
      throw IOError("file not found: " + std::string(path));
    raise();
  }
}

GZByteReader::~GZByteReader() {
  gzclose(f);
}

void GZByteReader::raise() {
  int err;
  throw IOError(gzerror(f, &err));
}

int GZByteReader::read(void *buf, int size) {
  int res = gzread(f, buf, size);
  if (res == 0 && !gzeof(f))
    raise();
  return res;
}

int GZByteReader::getc()
{
  int res = gzgetc(f);
  if (res == -1 && !gzeof(f))
    raise();
  return res;
}

/******************************************************************
 * Bit reader
 */

// This code may look unoptimized, like a few lookup tables or
// __bulitin_csz's would make a world of difference.  Don't be fooled.
// The fancier the optimization, the worse it will perform.  The
// succinct code here inlines very well, has excellent branch
// predictor behavior, and lends itself to compiler optimization.
// That being said, don't forget to compile with -O3.
class BitReader {
private:
  GZByteReader *r;

  uint8_t buf[4096];
  unsigned int bufBits;
  unsigned int bufBit;

  bool fillBuf();

public:
  bool eof;

  BitReader(GZByteReader *r);
  ~BitReader();

  int readBit();
  uint32_t readUnary();
  uint32_t readFixed(int w);
  uint32_t readSGamma();
  uint32_t readZeta(int k);
};

BitReader::BitReader(GZByteReader *r) :
    r(r), bufBits(0), bufBit(0), eof(false) {
}

BitReader::~BitReader() {
  delete r;
}

bool BitReader::fillBuf() {
  if (eof)
    return false;

  int len = r->read(buf, sizeof buf);
  bufBits = len * 8;
  bufBit = 0;
  eof = (bufBits == 0);
  return !eof;
}

int BitReader::readBit() {
  if (bufBit == bufBits && !fillBuf())
    return 0;

  // Return bit 7, then bit 6, and so on.  This mangles the
  // buffer, but is substantially faster than picking the right
  // bit out of it every time.
  int x = (bufBit++) / 8;
  int val = buf[x] >> 7;
  buf[x] <<= 1;
  return val;
}

uint32_t BitReader::readUnary() {
  uint32_t res = 0;

  while (!eof && !readBit())
    ++res;
  return res;
}

uint32_t BitReader::readFixed(int w) {
  uint32_t res = 0;

  while (w--) {
    res <<= 1;
    res |= readBit();
  }
  return res;
}

uint32_t BitReader::readSGamma() {
  uint32_t n = readUnary();
  return ((1 << n) | readFixed(n)) - 1;
}

uint32_t BitReader::readZeta(int k) {
  assert(k >= 1);

  uint32_t h = readUnary();
  uint32_t left = 1 << (h * k);
  uint32_t m = readFixed(h * k + k - 1);
  if (m < left)
    return m + left - 1;
  return (m << 1) + readBit() - 1;
}

/******************************************************************
 * Java properties reader
 */

class PropertiesReader {
private:
  typedef std::map<std::string, std::string> pmap;
  pmap props;

  GZByteReader *r;
  // See http://download.oracle.com/javase/6/docs/api/java/util/Properties.html#load(java.io.Reader)
  // for the (numerous) details on the properties file format.
  bool isLWS(char ch);
  bool isWS(char ch);
  bool physCR;
  int getPhys();
  bool natBOL;
  int getNat();
  int getEscaped(bool *escaped);
  bool getEntry();

public:
  PropertiesReader(const char *path);
  std::string get(const std::string &key, const std::string &def = "");
  int getInt(const std::string &key, int def = 0);
  void dump();
};

PropertiesReader::PropertiesReader(const char *path) {
  GZByteReader gzr(path);
  r = &gzr;

  physCR = false;
  natBOL = true;

  while (getEntry())
    ;

  r = NULL;
}

std::string PropertiesReader::get(const std::string &key,
    const std::string &def) {
  pmap::iterator it;
  it = props.find(key);
  if (it == props.end())
    return def;
  return it->second;
}

int PropertiesReader::getInt(const std::string &key, int def) {
  pmap::iterator it;
  it = props.find(key);
  if (it == props.end())
    return def;
  char *end;
  errno = 0;
  int res = strtol(it->second.c_str(), &end, 0);
  if (errno != 0 || *end != '\0')
    throw FormatError("malformed integer: " + it->second);
  return res;
}

void PropertiesReader::dump() {
  for (pmap::iterator it = props.begin(); it != props.end(); it++)
    std::cout << it->first << "=" << it->second << "\n";
}

bool PropertiesReader::isLWS(char ch) {
  return ch == ' ' || ch == '\t' || ch == '\f';
}

bool PropertiesReader::isWS(char ch) {
  return isLWS(ch) || ch == '\n' || ch == '\r';
}

// Get a character from a physical line.  This translates all line
// terminators into '\n'.
int PropertiesReader::getPhys() {
  int ch = r->getc();
  if (physCR && ch == '\n') {
    // This is the \n in a \r\n.  We already returned the
    // \n, so just skip it.  We have to do this early in
    // case we're in a \r\n\r\n-like sequence.
    ch = r->getc();
  }
  physCR = (ch == '\r');
  if (ch == '\r')
    ch = '\n';
  return ch;
}

// Get a character from a natural line.  This strips comments,
// beginning-of-line whitespace, and blanks.  After this, you should
// see just the properties, one after the next.
int PropertiesReader::getNat() {
  int ch = getPhys();

  while (natBOL) {
    // Consume whitespace
    while (isWS(ch))
      ch = getPhys();

    // Did we hit a comment?
    if (ch == '#' || ch == '!') {
      // Consume until EOL or EOF
      do {
        ch = getPhys();
      } while (ch != '\n' && ch != -1);
      if (ch == -1)
        return ch;
      // Loop and consume next line whitespace
    } else {
      // We hit real content
      natBOL = false;
      return ch;
    }
  }

  // We're not at the beginning of a line
  natBOL = (ch == '\n');
  return ch;
}

// Get a single character, decoding escape sequences and folding
// logical line continuation.  It's very convenient to do these two
// operations together, since they interact.
int PropertiesReader::getEscaped(bool *escaped) {
  next: int ch = getNat();
  if (ch == '\\') {
    *escaped = true;
    ch = getNat();
    switch (ch) {
    case 't':
      ch = '\t';
      break;
    case 'n':
      ch = '\n';
      break;
    case 'f':
      ch = '\f';
      break;
    case 'r':
      ch = '\r';
      break;
    case '\n':
      // Line continuation
      goto next;
    case 'u':
      throw FormatError("Java property file Unicode literals not implemented");
    }
  } else {
    *escaped = false;
  }
  return ch;
}

bool PropertiesReader::getEntry() {
  std::string key, val;
  int ch;
  bool escaped;

  // Get the key.  Read characters until we encounter a real =,
  // :, or WS.
  while (1) {
    ch = getEscaped(&escaped);
    if (!escaped && (ch == '=' || ch == ':' || isWS(ch) || ch == -1)) {
      // Got the entire key.  Skip trailing WS.
      while (!escaped && isLWS(ch))
        ch = getEscaped(&escaped);
      // If the first non-WS character is a real =
      // or :, consume it and any additional WS.
      if (!escaped && (ch == '=' || ch == ':')) {
        do {
          ch = getEscaped(&escaped);
        } while (!escaped && isLWS(ch));
      }
      break;
    }
    key += ch;
  }

  if (ch == -1 && key.length() == 0)
    // EOF
    return false;

  // Get the value.  Read characters until we encounter a real
  // newline.  Start with the character we already have.
  while (escaped || !(ch == '\n' || ch == -1)) {
    val += ch;
    ch = getEscaped(&escaped);
  }

  props[key] = val;
  return true;
}

/******************************************************************
 * WebGraph reader
 */

void Node::dump() const {
//        std::cout << node << " -> ";
  for (unsigned int i = 0; i < links.size(); ++i)
    std::cout << links[i] << " ";
  std::cout << "\n";
}

Reader::Reader(const std::string &basePath) :
    basePath(basePath), node(0) {
  // Read properties file
  PropertiesReader pr((basePath + ".properties").c_str());

  // Does it have an underlying graph?
  std::string graphBase = basePath;
  std::string under = pr.get("underlyinggraph");
  if (under.length()) {
    size_t slash = basePath.rfind('/');
    if (slash == basePath.npos)
      graphBase = "./";
    else
      graphBase = basePath.substr(0, slash + 1);
    graphBase += under;
    pr = PropertiesReader((graphBase + ".properties").c_str());
  }

  // Check the graph class
  std::string graphClass = pr.get("graphclass");
  if (!(graphClass == "it.unimi.dsi.webgraph.BVGraph"
      || graphClass == "class it.unimi.dsi.webgraph.BVGraph"))
    // We can't decode it
    throw FormatError("cannot decode unknown graph class " + graphClass);

  // Get properties
  windowSize = pr.getInt("windowsize");
  minIntervalLength = pr.getInt("minintervallength");
  zetaK = pr.getInt("zetak", 3);
  nodes = pr.getInt("nodes");
  std::string compressionFlags = pr.get("compressionflags");
  // We don't implement non-default compression flags
  if (!compressionFlags.empty())
    throw FormatError("non-default compression flags not implemented");

  // XXX Check for uncompressed .graph first?
  r = new BitReader(new GZByteReader((graphBase + ".graph.gz").c_str()));
  window = new Node[windowSize + 1];
}

Reader::~Reader() {
  delete r;
  delete[] window;
}

URLReader *Reader::newURLReader() {
  // XXX Check for uncompressed .urls first?
  return new URLReader(basePath + ".urls.gz");
}

int Reader::nat2int(uint32_t nat) {
  if (nat & 1)
    return -(int) ((nat + 1) / 2);
  return nat / 2;
}

template<class InputIterator, class OutputIterator>
OutputIterator checkedCopy(InputIterator first, InputIterator last,
    OutputIterator result, OutputIterator end) {
  assert(last - first <= end - result);
  return copy(first, last, result);
}

const Node *Reader::readNode() {
  Node *l = &window[node % (windowSize + 1)];
  l->node = node;
  ++node;

  // Out degree
  int degree = r->readSGamma(); // outdegreeCoding
  if (r->eof)
    return NULL;
  l->links.resize(degree);
  if (degree == 0)
    // If the degree is 0, nothing follows
    return l;

  Node::iterator out = l->links.begin(), end = l->links.end();

  // Read "reference" part
  int reference = 0;
  int nBlocks = 0;
  if (windowSize) {
    // If the window size is non-zero, the reference follows
    if ((reference = r->readUnary()) != 0) { // referenceCoding
      // If the reference is non-zero, the block
      // count and block list follow
      nBlocks = r->readSGamma(); // blockCountCoding
    }
  }
  uint32_t copyList[nBlocks];
  for (int i = 0; i < nBlocks; ++i)
    copyList[i] = r->readSGamma(); // blockCoding

  // Decode "reference" part
  if (reference) {
    // All blocks are decremented by one except the first
    for (int i = 1; i < nBlocks; ++i)
      ++copyList[i];

    assert(reference <= windowSize);
    Node *ref = &window[(l->node - reference) % (windowSize + 1)];

    // Copy bounded blocks
    Node::iterator src = ref->links.begin();
    for (int i = 0; i < nBlocks; ++i) {
      if (i % 2 == 0)
        out = checkedCopy(src, src + copyList[i], out, end);
      src += copyList[i];
    }

    // Copy final block if block count is even
    if (nBlocks % 2 == 0)
      out = checkedCopy(src, ref->links.end(), out, end);
  }

  if (out == end)
    // There is no "extra" part.  Since the reference part
    // was already sorted, we don't need to sort it here.
    return l;

  // Read intervals in "extra" part
  Node::iterator refEnd = out;
  int nIntervals = 0;
  if (minIntervalLength) {
    nIntervals = r->readSGamma();
  }
  uint32_t intervalLeft[nIntervals];
  uint32_t intervalLen[nIntervals];
  for (int i = 0; i < nIntervals; ++i) {
    intervalLeft[i] = r->readSGamma();
    intervalLen[i] = r->readSGamma();
  }

  // Decode intervals
  if (nIntervals)
    // The first interval left can be negative
    intervalLeft[0] = nat2int(intervalLeft[0]);
  // The first interval left if relative to the node number.
  // The -1 is a compensating factor for below.
  uint32_t prevRight = l->node - 1;
  for (int i = 0; i < nIntervals; ++i) {
    // The remaining interval lefts are relative to the
    // previous right + 1.
    int left = prevRight + intervalLeft[i] + 1;
    // Each length is relative to the minimum interval
    // length.
    int len = intervalLen[i] + minIntervalLength;
    prevRight = left + len;

    for (int j = 0; j < len; ++j)
      *(out++) = left + j;
  }

  // Sort the reference and interval parts.  Both parts are
  // individually sorted already.
  if (refEnd > l->links.begin() && refEnd != out)
    std::inplace_merge(l->links.begin(), refEnd, out);
  Node::iterator intEnd = out;

  // Read residuals
  int nResiduals = end - out;
  assert(nResiduals >= 0);
  int residuals[nResiduals];
  for (int i = 0; i < nResiduals; ++i)
    residuals[i] = r->readZeta(zetaK); // residualCoding

  // Decode residuals
  if (nResiduals)
    // The first residual can be negative
    residuals[0] = nat2int(residuals[0]);
  // The first residual is relative to the node number.  The -1
  // is a compensating factor.
  int residual = l->node - 1;
  for (int i = 0; i < nResiduals; ++i) {
    // The remaining residuals are relative to the
    // previous residual + 1.
    residual = residual + residuals[i] + 1;
    *(out++) = residual;
  }

  // Sort the reference+interval and residual parts.  Both parts
  // are individually sorted already.
  if (intEnd != l->links.begin() && intEnd != out)
    std::inplace_merge(l->links.begin(), intEnd, end);

  return l;
}

/******************************************************************
 * URL reader
 */

URLReader::URLReader(const std::string &path) {
  r = new GZByteReader(path.c_str());
  try {
    fill();
  } catch (...) {
    delete r;
    throw;
  }
}

URLReader::~URLReader() {
  delete r;
}

bool URLReader::fill() {
  bufLen = r->read(buf, sizeof buf);
  bufPos = 0;
  return bufLen > 0;
}

bool URLReader::readURL(std::string *out) {
  // Clear the output string.  At least in g++, this doesn't
  // free memory, so we won't be constantly reallocating if the
  // caller is reusing a string.
  out->clear();

  // Read to the next newline or EOF
  while (1) {
    // Get a chunk
    unsigned int start = bufPos;
    while (bufPos < bufLen && buf[bufPos] != '\n')
      ++bufPos;
    // Append to output
    out->append(&buf[start], bufPos - start);
    // Am I done?
    if (bufPos < bufLen) {
      // Skip the newline
      ++bufPos;
      // We definitely got data (even if it was a
      // blank line)
      return true;
    }
    // Read in more buffer
    if (!fill()) {
      // EOF.  We got data if there's anything in
      // the output string.
      return !out->empty();
    }
  }
}

}

/******************************************************************
 * main
 */

#if 0

// Print a graph in ASCIIGraph form
int main(int argc, char **argv)
{
  WebGraph::Reader r(argv[1]);

  const WebGraph::Node *n;
  std::cout << r.nodes << "\n";
  while ((n = r.readNode()))
  n->dump();
}

#endif
