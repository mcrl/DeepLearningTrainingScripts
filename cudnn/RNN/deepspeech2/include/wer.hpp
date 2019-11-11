#ifndef _WER_HPP_
#define _WER_HPP_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC double calc_wer(const char *chars, const char *target,
  int num_chars, int target_length);

#endif

