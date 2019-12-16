
#include <parser.hpp>

/* Constructor */
NPYParser::NPYParser() {

};

NPYParser::NPYParser(string filePath) {
	this->filePath = filePath;
};

string NPYParser::getFilePath() { return filePath; };

int NPYParser::load(void) { 
	if (filePath.empty()) {
		cerr << "Error! filePath is null" << endl;
		return -1;
	}
	ifstream input(filePath, ios::binary);
	if (!input.is_open()) {
		cerr << "Error! check your file path " << filePath << endl;
		return -1;
	}
	buffer.clear();
	vector<unsigned char> tmp_buffer(istreambuf_iterator<char>(input), {});
	buffer = tmp_buffer;
	input.close();
	return 0;
};

int NPYParser::load(string filePath) { 
	ifstream input(filePath, ios::binary);
	if (!input.is_open()) {
		cerr << "Error! check your file path " << filePath << endl;
		return -1;
	}
	vector<unsigned char> tmp_buffer(istreambuf_iterator<char>(input), {});
	buffer = tmp_buffer;
	input.close();
	return 0;
};

int NPYParser::_parse_header(void)
{
	const int magicString[] = {0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59};
	const int majorIdx = 6;
	const int minorIdx = 7;
	const int headerLengthIdx = 8;
	const int headerPreambleSize = 10;

	/* Magic String check */
	for (int i = 0; i < sizeof(magicString)/sizeof(int); ++i) {
		if(buffer[i] != magicString[i]) {
			cout << "magic number is wrong! (" << (int)buffer[i] << " : " << magicString[i] << ")" << endl;
			return -ERROR_MAGIC_STRING;
		}
	}

	//Format version
	header.majorVer = buffer[majorIdx];
	header.minorVer = buffer[minorIdx];

	//Get header length
	header.length = *(short*)&buffer[headerLengthIdx];
	header.dataIdx = header.length + headerPreambleSize; /* header length + preamble should be divided by 64 */

	//Descriptor
	/* ToDo */
	return 0;
};

int NPYParser::_parse_data(void)
{
	/* Refactoring */
	data.clear();
	int dataNum = (buffer.size() - header.dataIdx)/sizeof(float);
	for(int i = 0; i < dataNum; ++i) {
		data.push_back(*(float*)&buffer[header.dataIdx + (i * sizeof(float))]);
	}
	return 0;
};

int NPYParser::parse(void) {
	int ret = 0;
	checkParser(_parse_header());
	checkParser(_parse_data());
};

float* NPYParser::getDataStartAddr(void) {
	if(data.empty()) {
		cerr << "Warning : data is empty" << endl;
	}
	return &data[0];
}
