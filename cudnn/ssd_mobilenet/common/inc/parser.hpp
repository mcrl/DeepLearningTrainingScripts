#ifndef _PARSER_HPP_
#define _PARSER_HPP_
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <string>

#define ERROR_MAGIC_STRING 0x1
#define ERROR_MAGIC_NUMBER 0x2
#define NPY_PARSER_STATUS_SUCCESS 0x0

#define checkParser(expression)                              \
  {                                                          \
    int status = (expression);                               \
    if (status != NPY_PARSER_STATUS_SUCCESS) {               \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << (status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

using namespace std;
struct NPYHeader {
	/* ToDo : descrStr should be specified */
	string descrStr;
	int majorVer;
	int minorVer;
	bool fortranOrder;
	string shapeStr;
	int length;
	int dataIdx;
};

class NPYParser {
	public:
		NPYParser();
		NPYParser(string filePath);
		int load(void); 
		int load(string filePath); 
		int parse(void);
		string getFilePath(void);
		float* getDataStartAddr(void);
	private:
		int _parse_header(void);
		int _parse_data(void);

		struct NPYHeader header;
		string filePath;
		int shapeDim;
		vector<int> shapeArray;
		vector<unsigned char> buffer;
		vector<float>data;
};
#endif
