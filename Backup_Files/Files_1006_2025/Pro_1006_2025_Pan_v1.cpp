// =====================================================
// PT-ACTIT: GP + GA Hybrid Image Filter Optimization
// Based on Fujishima & Nagao (2005)
// =====================================================

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

// =====================================================
// Configuration
// =====================================================
#define POP_SIZE 50
#define GENERATIONS 200
#define OFFSPRING_COUNT 16
#define MUTATION_RATE 0.9
#define MAX_DEPTH 6
#define ENABLE_GA true
#define GA_POP 20
#define GA_GENERATIONS 40
#define INITIAL_BIAS_THRESHOLD 0.05
#define BIAS_DECAY 0.9
#define BIAS_WINDOW 5

// =====================================================
// Random utilities
// =====================================================
random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> prob(0.0, 1.0);
uniform_int_distribution<> randint(0, 999999);

// =====================================================
// Filter types
// =====================================================
enum FilterType {
    TERMINAL_INPUT,
    GAUSSIAN_BLUR, MED_BLUR, BLUR, BILATERAL_FILTER,
    SOBEL_X, SOBEL_Y, CANNY,
    THRESHOLD_31, THRESHOLD_63, THRESHOLD_127,
    ERODE, DILATE,
    BITWISE_AND, BITWISE_OR, BITWISE_NOT, BITWISE_XOR,
    DIFF_PROCESS
};

struct ParamDesc {
    int n;
    double minv;
    double maxv;
};
unordered_map<FilterType, ParamDesc> g_paramDesc;

// 初始化各滤波器参数范围
void initParamDesc() {
    g_paramDesc[GAUSSIAN_BLUR] = { 2, 1.0, 31.0 };
    g_paramDesc[MED_BLUR] = { 1, 1.0, 31.0 };
    g_paramDesc[BLUR] = { 1, 1.0, 31.0 };
    g_paramDesc[BILATERAL_FILTER] = { 3, 1.0, 150.0 };
    g_paramDesc[CANNY] = { 2, 1.0, 255.0 };
    g_paramDesc[THRESHOLD_31] = { 1, 0.0, 255.0 };
    g_paramDesc[THRESHOLD_63] = { 1, 0.0, 255.0 };
    g_paramDesc[THRESHOLD_127] = { 1, 0.0, 255.0 };
    g_paramDesc[ERODE] = { 1, 0.0, 5.0 };
    g_paramDesc[DILATE] = { 1, 0.0, 5.0 };
}

// =====================================================
// Tree definition
// =====================================================
struct TreeNode {
    FilterType type;
    vector<shared_ptr<TreeNode>> children;
    vector<double> params;
};

shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
    if (!node) return nullptr;
    auto newNode = make_shared<TreeNode>();
    newNode->type = node->type;
    newNode->params = node->params;
    for (auto& c : node->children) newNode->children.push_back(cloneTree(c));
    return newNode;
}

// =====================================================
// Gray code utilities
// =====================================================
inline int binaryToGray(int num) { return num ^ (num >> 1); }
inline int grayToBinary(int num) {
    for (int mask = num >> 1; mask != 0; mask >>= 1) num ^= mask;
    return num;
}
inline string intToBits(int n, int bits = 8) {
    string s(bits, '0');
    for (int i = bits - 1; i >= 0; --i) s[i] = (n & 1) + '0', n >>= 1;
    return s;
}
inline int bitsToInt(const string& s) {
    int val = 0;
    for (char c : s) val = (val << 1) | (c - '0');
    return val;
}
inline string grayEncode(double val, double minv, double maxv, int bits = 8) {
    double norm = (val - minv) / (maxv - minv);
    norm = max(0.0, min(1.0, norm));
    int bin = int(norm * ((1 << bits) - 1));
    return intToBits(binaryToGray(bin), bits);
}
inline double grayDecode(const string& gray, double minv, double maxv, int bits = 8) {
    int bin = grayToBinary(bitsToInt(gray));
    double norm = double(bin) / ((1 << bits) - 1);
    return minv + norm * (maxv - minv);
}
inline string mutateGrayBits(string s, double rate = 0.01) {
    for (auto& c : s)
        if (prob(rng) < rate) c = (c == '0' ? '1' : '0');
    return s;
}

// =====================================================
// Random Tree generation
// =====================================================
shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = MAX_DEPTH) {
    if (depth >= maxDepth || prob(rng) < 0.1) {
        auto t = make_shared<TreeNode>();
        t->type = TERMINAL_INPUT;
        return t;
    }
    FilterType t = static_cast<FilterType>(1 + (rng() % (BITWISE_XOR - 1)));
    auto node = make_shared<TreeNode>();
    node->type = t;

    if (g_paramDesc.count(t)) {
        auto pd = g_paramDesc[t];
        node->params.resize(pd.n);
        for (int i = 0; i < pd.n; i++) {
            uniform_real_distribution<> ud(pd.minv, pd.maxv);
            node->params[i] = ud(rng);
        }
    }
    int numChildren = (t == BITWISE_AND || t == BITWISE_OR || t == BITWISE_XOR || t == DIFF_PROCESS) ? 2 : 1;
    for (int i = 0; i < numChildren; i++) node->children.push_back(generateRandomTree(depth + 1, maxDepth));
    return node;
}

// =====================================================
// Execute filter tree
// =====================================================
Mat executeTree(shared_ptr<TreeNode> node, const Mat& input) {
    if (!node) return input;
    switch (node->type) {
    case TERMINAL_INPUT:
        return input.clone();
    case GAUSSIAN_BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        GaussianBlur(executeTree(node->children[0], input), dst, Size(k, k), node->params[1]);
        return dst;
    }
    case MED_BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        medianBlur(executeTree(node->children[0], input), dst, k);
        return dst;
    }
    case BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        blur(executeTree(node->children[0], input), dst, Size(k, k));
        return dst;
    }
    case BILATERAL_FILTER: {
        Mat dst;
        bilateralFilter(executeTree(node->children[0], input), dst,
            int(node->params[0]), node->params[1], node->params[2]);
        return dst;
    }
    case SOBEL_X: {
        Mat dst, tmp = executeTree(node->children[0], input);
        Sobel(tmp, dst, CV_8U, 1, 0);
        return dst;
    }
    case SOBEL_Y: {
        Mat dst, tmp = executeTree(node->children[0], input);
        Sobel(tmp, dst, CV_8U, 0, 1);
        return dst;
    }
    case CANNY: {
        Mat dst, tmp = executeTree(node->children[0], input);
        Canny(tmp, dst, node->params[0], node->params[1]);
        return dst;
    }
    case THRESHOLD_31:
    case THRESHOLD_63:
    case THRESHOLD_127: {
        Mat dst, tmp = executeTree(node->children[0], input);
        threshold(tmp, dst, node->params[0], 255, THRESH_BINARY);
        return dst;
    }
    case ERODE: {
        Mat dst, tmp = executeTree(node->children[0], input);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + 2 * int(node->params[0]), 1 + 2 * int(node->params[0])));
        erode(tmp, dst, kernel);
        return dst;
    }
    case DILATE: {
        Mat dst, tmp = executeTree(node->children[0], input);
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + 2 * int(node->params[0]), 1 + 2 * int(node->params[0])));
        dilate(tmp, dst, kernel);
        return dst;
    }
    case BITWISE_AND: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        bitwise_and(a, b, dst);
        return dst;
    }
    case BITWISE_OR: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        bitwise_or(a, b, dst);
        return dst;
    }
    case BITWISE_XOR: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        bitwise_xor(a, b, dst);
        return dst;
    }
    case BITWISE_NOT: {
        Mat a = executeTree(node->children[0], input);
        Mat dst;
        bitwise_not(a, dst);
        return dst;
    }
    case DIFF_PROCESS: {
        Mat a = executeTree(node->children[0], input);
        Mat b = executeTree(node->children[1], input);
        Mat dst;
        absdiff(a, b, dst);
        return dst;
    }
    default:
        return input.clone();
    }
}

// =====================================================
// Fitness calculation (MSE similarity)
// =====================================================
double calcFitness(shared_ptr<TreeNode> tree, const Mat& in, const Mat& target) {
    Mat out = executeTree(tree, in);
    if (out.size() != target.size()) resize(out, out, target.size());
    Mat diff;
    absdiff(out, target, diff);
    return 1.0 - mean(diff)[0] / 255.0;
}

// =====================================================
// Bias calculation
// =====================================================
double calcBias(const vector<double>& hist, int window = BIAS_WINDOW) {
    if (hist.size() < window + 1) return 1.0;
    double recentAvg = 0.0;
    for (int i = hist.size() - window; i < (int)hist.size(); ++i) recentAvg += hist[i];
    recentAvg /= window;
    double totalAvg = accumulate(hist.begin(), hist.end(), 0.0) / hist.size();
    double bias = fabs(recentAvg - totalAvg) / (totalAvg + 1e-9);
    return bias;
}

// =====================================================
// GA using Gray encoding
// =====================================================
vector<double> runGrayGA_forTree(shared_ptr<TreeNode> tree, const Mat& in, const Mat& target) {
    vector<shared_ptr<TreeNode>> paramNodes;
    function<void(shared_ptr<TreeNode>)> collect = [&](shared_ptr<TreeNode> n) {
        if (!n) return;
        if (!n->params.empty()) paramNodes.push_back(n);
        for (auto& c : n->children) collect(c);
        };
    collect(tree);
    if (paramNodes.empty()) return {};

    vector<pair<double, double>> bounds;
    vector<string> baseGenes;
    for (auto& n : paramNodes) {
        auto pd = g_paramDesc[n->type];
        for (int i = 0; i < pd.n; i++) {
            bounds.push_back({ pd.minv, pd.maxv });
            baseGenes.push_back(grayEncode(n->params[i], pd.minv, pd.maxv, 8));
        }
    }
    int geneLen = baseGenes.size();

    struct GrayInd {
        vector<string> genes;
        double fit;
    };
    vector<GrayInd> pop(GA_POP);

    for (auto& ind : pop) {
        ind.genes = baseGenes;
        for (auto& g : ind.genes) g = mutateGrayBits(g, 0.05);
        vector<double> decoded;
        for (int i = 0; i < geneLen; i++) decoded.push_back(grayDecode(ind.genes[i], bounds[i].first, bounds[i].second));
        auto treeClone = cloneTree(tree);
        vector<shared_ptr<TreeNode>> pn;
        collect(treeClone);
        int pos = 0;
        for (auto& n : pn)
            for (auto& p : n->params) p = decoded[pos++];
        ind.fit = calcFitness(treeClone, in, target);
    }

    for (int gen = 0; gen < GA_GENERATIONS; gen++) {
        sort(pop.begin(), pop.end(), [](auto& a, auto& b) { return a.fit > b.fit; });
        vector<GrayInd> newpop;
        newpop.push_back(pop[0]);
        while ((int)newpop.size() < GA_POP) {
            int a = rng() % GA_POP, b = rng() % GA_POP;
            GrayInd child = pop[a];
            for (int i = 0; i < geneLen; i++) {
                if (prob(rng) < 0.5) child.genes[i] = pop[b].genes[i];
                child.genes[i] = mutateGrayBits(child.genes[i], 0.01);
            }
            vector<double> decoded;
            for (int i = 0; i < geneLen; i++) decoded.push_back(grayDecode(child.genes[i], bounds[i].first, bounds[i].second));
            auto treeClone = cloneTree(tree);
            vector<shared_ptr<TreeNode>> pn;
            collect(treeClone);
            int pos = 0;
            for (auto& n : pn)
                for (auto& p : n->params) p = decoded[pos++];
            child.fit = calcFitness(treeClone, in, target);
            newpop.push_back(child);
        }
        pop.swap(newpop);
    }

    sort(pop.begin(), pop.end(), [](auto& a, auto& b) { return a.fit > b.fit; });
    vector<double> decoded;
    for (int i = 0; i < geneLen; i++) decoded.push_back(grayDecode(pop.front().genes[i], bounds[i].first, bounds[i].second));
    return decoded;
}

// =====================================================
// GP+GA main process
// =====================================================
void runPT_ACTIT(const Mat& in, const Mat& target) {
    initParamDesc();
    vector<shared_ptr<TreeNode>> population;
    for (int i = 0; i < POP_SIZE; i++) population.push_back(generateRandomTree());
    vector<double> eliteHist;
    double biasThreshold = INITIAL_BIAS_THRESHOLD;

    for (int gen = 0; gen < GENERATIONS; gen++) {
        vector<double> scores(POP_SIZE);
        for (int i = 0; i < POP_SIZE; i++) scores[i] = calcFitness(population[i], in, target);
        auto bestIt = max_element(scores.begin(), scores.end());
        double eliteScore = *bestIt;
        int eliteIdx = distance(scores.begin(), bestIt);
        eliteHist.push_back(eliteScore);
        double bias = calcBias(eliteHist);
        cout << "Gen " << gen + 1 << " Elite=" << eliteScore << " Bias=" << bias << endl;

        // ---- Trigger GA when Bias low ----
        if (ENABLE_GA && bias < biasThreshold) {
            cout << "[PT-ACTIT] Trigger GA Phase (Bias=" << bias << ")" << endl;
            auto best = runGrayGA_forTree(population[eliteIdx], in, target);
            if (!best.empty()) {
                vector<shared_ptr<TreeNode>> pn;
                function<void(shared_ptr<TreeNode>)> collect = [&](shared_ptr<TreeNode> n) {
                    if (!n) return;
                    if (!n->params.empty()) pn.push_back(n);
                    for (auto& c : n->children) collect(c);
                    };
                collect(population[eliteIdx]);
                int pos = 0;
                for (auto& n : pn)
                    for (auto& p : n->params) p = best[pos++];
            }
            biasThreshold *= BIAS_DECAY;
        }

        // ---- Simple GP mutation ----
        for (int i = 0; i < POP_SIZE; i++) {
            if (prob(rng) < MUTATION_RATE) {
                population[i] = generateRandomTree();
            }
        }
    }
}

// =====================================================
// main
// =====================================================
int main() {
    cout << "=== PT-ACTIT Hybrid GP+GA System ===" << endl;
    Mat input = imread("./input.png", IMREAD_GRAYSCALE);
    Mat target = imread("./target.png", IMREAD_GRAYSCALE);
    if (input.empty() || target.empty()) {
        cerr << "Error: missing ./input.png or ./target.png" << endl;
        return -1;
    }
    runPT_ACTIT(input, target);
    return 0;
}

/*
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define sysRunTimes 3
#define numSets 8 // the num of sets(pairs)
#define idSet 1 // for mark the selected set if the numSets been set of 1
#define POP_SIZE 50
#define GENERATIONS 200
#define OFFSPRING_COUNT 16
#define MUTATION_RATE 0.9
#define NUM_TYPE_FUNC 19
#define MAX_DEPTH 10 // { 0, 1, 2, ... }
#define ENABLE_GA true
#define GA_POP 20
#define GA_GENERATIONS 40
#define INITIAL_BIAS_THRESHOLD 0.05
#define BIAS_DECAY 0.9
#define BIAS_WINDOW 5

random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> prob(0.0, 1.0);

enum FilterType { // type-terminal and type-function
    TERMINAL_INPUT,
    GAUSSIAN_BLUR,
    MED_BLUR,
    BLUR,
    BILATERAL_FILTER,
    SOBEL_X,
    SOBEL_Y,
    CANNY,
    THRESHOLD,
    ERODE,
    DILATE,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_NOT,
    BITWISE_XOR,
    DIFF_PROCESS,
    CON_PRO_SINGLE_TIME,
};

struct ParamDesc {
    int n;
    double minv;
    double maxv;
};
unordered_map<FilterType, ParamDesc> g_paramDesc;

void initParamDesc() {
    g_paramDesc[GAUSSIAN_BLUR] = { 2, 1.0, 31.0 };
    g_paramDesc[MED_BLUR] = { 1, 1.0, 31.0 };
    g_paramDesc[BLUR] = { 1, 1.0, 31.0 };
    g_paramDesc[BILATERAL_FILTER] = { 3, 1.0, 150.0 };
    g_paramDesc[CANNY] = { 2, 1.0, 255.0 };
    g_paramDesc[THRESHOLD] = { 1, 0.0, 255.0 };
    g_paramDesc[ERODE] = { 1, 0.0, 5.0 };
    g_paramDesc[DILATE] = { 1, 0.0, 5.0 };
    g_paramDesc[CON_PRO_SINGLE_TIME] = { 3, 0.0, 15.0 };
}

struct TreeNode {
    FilterType type;
    vector<shared_ptr<TreeNode>> children;
    vector<double> params;
};

struct genType {
    shared_ptr<TreeNode> eliteTree;
    double eliteFValue;
    double genMinFValue;
    double genAveFValue;
    double genDevFValue;
};

// for storing the f-value of every individual in the group
double indFValInfo[POP_SIZE][numSets + 1];
int curMaxFvalIdx = 0;
double curThreshFVal = 3.00;

void imgShow(const string& name, const Mat& img) {
    imshow(name, img);
    waitKey(0);
    destroyAllWindows();
}

shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
    if (!node) return nullptr;
    auto newNode = make_shared<TreeNode>();
    newNode->type = node->type;
    newNode->params = node->params;
    for (auto& c : node->children) newNode->children.push_back(cloneTree(c));
    return newNode;
}

// =====================================================
// Gray code utilities
// =====================================================

/*
  Turn bin to gray -> Convert a regular integer into an integer that can be interpreted as a Gray code.
  Eg: 3 -> 0011(bin) -> 0010(gray) -> 2
*/
inline int binaryToGray(int num) { return num ^ (num >> 1); }

/*
  Turn gray to bin -> Convert an integer that can be interpreted as a Gray code into a regular integer.
  Eg: 2 -> 0010(gray) -> 0011(bin) -> 3
*/
inline int grayToBinary(int num) {
    for (int mask = num >> 1; mask != 0; mask >>= 1) num ^= mask;
    return num;
}

/*
  Convert the integer n into a binary string of length bits.
  intToBits(5, 8) → "00000101"
*/
inline string intToBits(int n, int bits = 8) {
    string s(bits, '0');
    for (int i = bits - 1; i >= 0; --i) s[i] = (n & 1) + '0', n >>= 1;
    return s;
}

/*
  Convert a binary string of length bits into the integer n
  "00000101" → intToBits(5, 8)
*/
inline int bitsToInt(const string& s) {
    int val = 0;
    for (char c : s) val = (val << 1) | (c - '0');
    return val;
}

/*
  Map a real-valued parameter (e.g., σ = 1.2) to a Gray-coded binary string.
  -> (1.2 - 0.0)/(5.0 - 0.0) = 0.24
  -> 0.24 * 255(2^8 - 1) ≈ 61
  -> binaryToGray(61) → 111101 → 100011(gray) → 35(gray)
  -> intToBits(35, 8) → "00100011"
*/
inline string grayEncode(double val, double minv, double maxv, int bits = 8) {
    double norm = (val - minv) / (maxv - minv);
    norm = max(0.0, min(1.0, norm));
    int bin = int(norm * ((1 << bits) - 1));
    return intToBits(binaryToGray(bin), bits);
}

/*
  Decode the Gray code back into the corresponding real-valued parameter.
  -> grayDecode("00111101", 0.0, 5.0)　→　61(bin)　
  　 →　norm = 61 / 255 ≈ 0.239　→　val = 0 + 0.239*5 = 1.195
*/
inline double grayDecode(const string& gray, double minv, double maxv, int bits = 8) {
    int bin = grayToBinary(bitsToInt(gray));
    double norm = double(bin) / ((1 << bits) - 1);
    return minv + norm * (maxv - minv);
}

/*
  Randomly flip certain bits in the Gray-coded string with a probability of rate (genetic mutation).
  "00110110(gray)" -> "00111110(gray)"
*/
inline string mutateGrayBits(string s, double rate = 0.01) {
    for (auto& c : s)
        if (prob(rng) < rate) c = (c == '0' ? '1' : '0');
    return s;
}

shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = MAX_DEPTH) {
    if (depth >= maxDepth || prob(rng) < 0.1) {
        auto t = make_shared<TreeNode>();
        t->type = TERMINAL_INPUT;
        return t;
    }
    FilterType t = static_cast<FilterType>(1 + (rng() % NUM_TYPE_FUNC));
    auto node = make_shared<TreeNode>();
    node->type = t;

    if (g_paramDesc.count(t)) {
        // pd -> { numParam, minVal, maxVal }
        auto pd = g_paramDesc[t];
        node->params.resize(pd.n);
        for (int i = 0; i < pd.n; i++) {
            // Create a random number generator that returns values uniformly distributed over the interval [minv, maxv].
            uniform_real_distribution<> ud(pd.minv, pd.maxv);
            node->params[i] = ud(rng);
        }
    }
    int numChildren = (t == BITWISE_AND || t == BITWISE_OR || t == BITWISE_XOR || t == DIFF_PROCESS) ? 2 : 1;
    for (int i = 0; i < numChildren; i++) node->children.push_back(generateRandomTree(depth + 1, maxDepth));
    return node;
}

Mat executeTree(shared_ptr<TreeNode> node, const Mat& input) {
    if (!node) return input;
    switch (node->type) {
    case TERMINAL_INPUT:
        return input.clone();
    case GAUSSIAN_BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        GaussianBlur(executeTree(node->children[0], input), dst, Size(k, k), node->params[1]);
        return dst;
    }
    case MED_BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        medianBlur(executeTree(node->children[0], input), dst, k);
        return dst;
    }
    case BLUR: {
        int k = int(node->params[0]) | 1;
        Mat dst;
        blur(executeTree(node->children[0], input), dst, Size(k, k));
        return dst;
    }
    case BILATERAL_FILTER: {
        Mat dst;
        bilateralFilter(executeTree(node->children[0], input), dst,
            int(node->params[0]), node->params[1], node->params[2]);
        return dst;
    }
    case SOBEL_X: {
        Mat dst;
        Sobel(executeTree(node->children[0], input), dst, CV_8U, 1, 0);
        return dst;
    }
    case SOBEL_Y: {
        Mat dst;
        Sobel(executeTree(node->children[0], input), dst, CV_8U, 0, 1);
        return dst;
    }
    case CANNY: {
        Mat dst;
        Canny(executeTree(node->children[0], input), dst, node->params[0], node->params[1]);
        return dst;
    }
    case THRESHOLD: {
        Mat dst;
        threshold(executeTree(node->children[0], input), dst, node->params[0], 255, THRESH_BINARY);
        return dst;
    }
    case ERODE: {
        Mat dst;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + 2 * int(node->params[0]), 1 + 2 * int(node->params[0])));
        erode(executeTree(node->children[0], input), dst, kernel);
        return dst;
    }
    case DILATE: {
        Mat dst;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(1 + 2 * int(node->params[0]), 1 + 2 * int(node->params[0])));
        dilate(executeTree(node->children[0], input), dst, kernel);
        return dst;
    }
    case BITWISE_AND: {
        Mat dst;
        bitwise_and(executeTree(node->children[0], input), executeTree(node->children[1], input), dst);
        return dst;
    }
    case BITWISE_OR: {
        Mat dst;
        bitwise_or(executeTree(node->children[0], input), executeTree(node->children[1], input), dst);
        return dst;
    }
    case BITWISE_XOR: {
        Mat dst;
        bitwise_xor(executeTree(node->children[0], input), executeTree(node->children[1], input), dst);
        return dst;
    }
    case BITWISE_NOT: {
        Mat dst;
        bitwise_not(executeTree(node->children[0], input), dst);
        return dst;
    }
    case DIFF_PROCESS: {
        int absoluteFlag = 0;
        Mat dst = Mat::zeros(Size(input.cols, input.rows), CV_8UC1);
        for (int j = 0; j < input.rows; j++)
        {
            for (int i = 0; i < input.cols; i++) {
                int diffVal = executeTree(node->children[0], input).at<uchar>(j, i) - executeTree(node->children[1], input).at<uchar>(j, i);
                if (diffVal < 0) {
                    if (absoluteFlag != 0) {
                        diffVal = abs(diffVal);
                    }
                    else {
                        diffVal = 0;
                    }
                }
                dst.at<uchar>(j, i) = diffVal;
            }
        }
        return dst;
    }
    case CON_PRO_SINGLE_TIME: {
        Mat dst = Mat::zeros(input.size(), CV_8UC1);
        Mat maskImg = executeTree(node->children[0], input).clone();
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        for (int idxET = 0; idxET < (int)(node->params[0]); idxET++) {
            erode(maskImg, maskImg, kernel);
        }
        vector<vector<Point>> contours;
        findContours(maskImg, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        Mat mask = Mat::zeros(maskImg.size(), CV_8UC1);
        for (const auto& contour : contours) {
            Rect bounding_box = boundingRect(contour);
            double aspect_ratio = static_cast<double>(bounding_box.width) / bounding_box.height;
            if ((aspect_ratio <= (1 - (int)(node->params[1]) * 0.1) || aspect_ratio > (1 + (int)(node->params[1]) * 0.1)) && cv::contourArea(contour) < 100 * (int)(node->params[2])) {
                drawContours(mask, vector<vector<Point>>{contour}, -1, Scalar(255), -1);
            }
        }
        for (int y = 0; y < dst.rows; y++) {
            for (int x = 0; x < dst.cols; x++) {
                if (mask.at<uchar>(y, x) == 255) {
                    dst.at<uchar>(y, x) = 255;
                }
            }
        }
        return dst;
    }
    default:
        return input.clone();
    }
}
*/

