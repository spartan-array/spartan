// Copyright (c) 2008, Google Inc.
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// ---
// This file is a compatibility layer that defines Google's version of
// command line flags that are used for configuration.
//
// We put flags into their own namespace.  It is purposefully
// named in an opaque way that people should have trouble typing
// directly.  The idea is that DEFINE puts the flag in the weird
// namespace, and DECLARE imports the flag from there into the
// current namespace.  The net result is to force people to use
// DECLARE to get access to a flag, rather than saying
//   extern bool FLAGS_logtostderr;
// or some such instead.  We want this so we can put extra
// functionality (like sanity-checking) in DECLARE if we want,
// and make sure it is picked up everywhere.
//
// We also put the type of the variable in the namespace, so that
// people can't DECLARE_int32 something that they DEFINE_bool'd
// elsewhere.
#ifndef BASE_COMMANDLINEFLAGS_H__
#define BASE_COMMANDLINEFLAGS_H__

#include "glog-config.h"
#include <string>
#include <string.h>               // for memchr
#include <stdlib.h>               // for getenv

#include <gflags/gflags.h>
#endif  // BASE_COMMANDLINEFLAGS_H__
