#include "wer.hpp"

#include <stdio.h>

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

double calc_wer(const char *chars, const char *target,
  int num_chars, int target_length)
{
  std::map<std::string, int> words;

  std::string _chars(chars);
  std::string _target(target);
  std::string word;

  std::stringstream ss_chars(_chars);
  std::stringstream ss_target(_target);

  std::vector<int> items_chars;
  std::vector<int> items_target;

  int idx = 0;

  while (ss_chars >> word) {
    if (words.find(word) == words.end()) {
      words[word] = idx++;
    }
    items_chars.push_back(words[word]);
  }

  while (ss_target >> word) {
    if (words.find(word) == words.end()) {
      words[word] = idx++;
    }
    items_target.push_back(words[word]);
  }

  std::vector<std::vector<int>> table(items_chars.size() + 1,
    std::vector<int>(items_target.size() + 1, 0));

  for (int i = 0; i <= items_chars.size(); i++) table[i][0] = i;
  for (int i = 0; i <= items_target.size(); i++) table[0][i] = i;

  for (int i = 1; i <= items_chars.size(); i++) {
    for (int j = 1; j <= items_target.size(); j++) {
      if (items_chars[i - 1] == items_target[j - 1]) {
        table[i][j] = table[i - 1][j - 1];
      } else {
        table[i][j] = std::min(table[i - 1][j - 1],
          std::min(table[i][j - 1], table[i - 1][j])) + 1;
      }
    }
  }

  return 1.0 * table.back().back() / items_target.size();
}

