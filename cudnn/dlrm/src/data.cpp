
#include "data.h"


using namespace std;

vector<string> featureList[50];
int getIdx(vector<string>& sv, string &s) {
    int res = lower_bound(sv.begin(), sv.end(), s) - sv.begin();
    return res;
}

Data::Data () {
}

Data::Data (string s, int processed) {
    vector<string> sv;
    if( processed ) sv = split(s, ' ');
    else sv = split(s, '\t');

    res = atoi(sv[0].c_str());
    for (int i = 0; i < 13; i++) {
        dense[i] = atoi(sv[i+1].c_str());
    }
    if( processed ) {
        for (int i = 0; i < 26; i++) {
            processed_sparse[i] = atoi(sv[i+14].c_str());
        }
    }
    else {
        for (int i = 0; i < 26; i++) {
            sparse[i] = sv[i+14];
            processed_sparse[i] = getIdx(featureList[14+i], sparse[i]);
        }
    }
}

Data::~Data () {
}

// Output preprocessed data to file
void Data::output (ofstream& out) {
    out << res << " ";
    for (int i = 0; i < 13; i++) {
        out << dense[i] << " ";
    }
    for (int i = 0; i < 26; i++) {
        out << processed_sparse[i];
        string c = " ";
        if( i == 25 ) c = "\n";
        out << c;
    }
}

void Data::denseToArray(float *out) {
    for (int i = 0; i < 13; i++) {
        float x = dense[i] < 0 ? 0.0 : (float) dense[i];
        out[i] = log(x + 1);
    }
}

void Data::sparseToArray3(int *out) {
    for (int i = 0; i < 26; i++) {
        out[i] = processed_sparse[i];
    }
}

void Data::sparseToArray(int *out[], int batch) {
    for (int i = 0; i < 26; i++) {
        out[i][batch] = processed_sparse[i];
    }
}

void Data::sparseToBagArray(int *out[], int batch) {
    for (int i = 0; i < 26; i++) {
        out[i][batch * bag_size] = processed_sparse[i];
        for (int j = 1; j < bag_size; j++) out[i][batch * bag_size + j] = -1;
    }
}

void Data::serialize (int *out) {
    out[0] = res;
    for (int i = 0; i < 13; i++) out[i + 1] = dense[i];
    for (int i = 0; i < 26; i++) out[i + 1 + 13] = processed_sparse[i];
}

void Data::deserialize (int *in) {
    res = in[0];
    for (int i = 0; i < 13; i++) dense[i] = in[i + 1];
    for (int i = 0; i < 26; i++) processed_sparse[i] = in[i + 1 + 13];
}


inline bool file_exists (const char *fname) {
    ifstream f(fname);
    return f.good();
}

/////////////////////////////////////////////////////////////////////
//                     Read and Preprocess data                    //
/////////////////////////////////////////////////////////////////////
void data_preprocess () {
    cout << "[Preprocess raw data]\n";
    ifstream dataFile("./data/data.txt");
    ofstream processedFile("./data/processed.txt");
    ofstream featureMapFile("./data/feature_map.txt");

    string s;
    int cnt = 0;
    while ( getline(dataFile, s) ) {
        cout << "\rReading raw data line: " << ++cnt << "/" << DATA_FILE_LINES << std::flush;
        // if( cnt > DATA_FILE_LINES / 10 ) break;
        vector<string> sv = split(s, '\t');
        for (int i = 0; i < sv.size(); i++) {
            featureList[i].push_back(sv[i]);
        }

    }
    cout << '\n';

    for (int i = 0; i < 40; i++){ // 0 is answer
        cout << "Compressing features: " << i << "/" << 40 << std::flush;
        if( i < 14 ) { // skip dense features
            featureMapFile << "\n";
            continue;
        }

        featureList[i].push_back("");
        sort(featureList[i].begin(), featureList[i].end());
        featureList[i].erase(unique(featureList[i].begin(), featureList[i].end()), featureList[i].end());
        for (int j = 0; j < featureList[i].size(); j++) {
            featureMapFile << featureList[i][j];
            string c = " ";
            if( j == featureList[i].size() - 1 ) c = "\n";
            featureMapFile << c;
        }

        cout << " #=" << featureList[i].size() << "\n" << std::flush; 
    }
    cout << "Compressing features Done!\n";

    dataFile.clear();
    dataFile.seekg(0, ios::beg);

    cnt = 0;
    while ( getline(dataFile, s) ) {
        cout << "\rProcessing data line: " << ++cnt << "/" << DATA_FILE_LINES << std::flush;
        Data data(s, 0);
        data.output(processedFile);
    }
    cout << "\n";
}


/////////////////////////////////////////////////////////////////////
//                        Load preprocessed data                   //
/////////////////////////////////////////////////////////////////////
void data_load (int numTrain, int numTest, vector<Data>& train_data, vector<Data>& test_data, int *numFeatures) {
    if ( !file_exists("./data/processed.txt") ) {
        cout << "processed file not found\n" << std::flush; 
        data_preprocess();
    }
    else cout << "processed file found\n" << std::flush;

    cout << "[Load processed data]\n" << std::flush;
    cout << numTrain << " training data\n" << std::flush;
    cout << numTest << " test data\n" << std::flush;

    // Load sparse feature list
    string s;
    int cnt = 0;
    ifstream featureMapFile("./data/feature_map.txt");
    while ( getline(featureMapFile, s) ) {
        cnt++;
        if( cnt >= 15 ) featureList[cnt-15] = split(s, ' ');
    }
    for (int i = 0; i < 26; i++) numFeatures[i] = featureList[i].size();

    // TODO: shuffle data
    cnt = 0;
    ifstream processedFile("./data/processed.txt");
    while ( getline(processedFile, s) ) {
        cout << "\rReading processed data: " << ++cnt << "/" << DATA_FILE_LINES << std::flush;
        if( test_data.size() < numTest ) test_data.push_back(Data(s, 1));
        else if( train_data.size() < numTrain ) train_data.push_back(Data(s, 1));
        else break;
    }
    cout << '\n';

    random_shuffle(train_data.begin(), train_data.end());
    random_shuffle(test_data.begin(), test_data.end());
}
