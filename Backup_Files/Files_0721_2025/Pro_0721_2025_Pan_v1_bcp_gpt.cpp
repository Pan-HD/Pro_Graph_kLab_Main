// actit_opencv_template.cpp
// ACTIT: 木構造状画像変換の自動構築法（C++ & OpenCV 正确实现MGG）

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

//----------------------------------------
// 滤波器种类
//----------------------------------------
enum FilterType {
    TERMINAL_INPUT,
    MEAN,
    MIN,
    BOUNDED_SUM,
};

//----------------------------------------
// 树节点结构
//----------------------------------------
struct TreeNode {
    FilterType type;
    vector<shared_ptr<TreeNode>> children;
};

//----------------------------------------
// 滤波器函数实现
//----------------------------------------
Mat meanFilter(const Mat& img) {
    Mat out;
    blur(img, out, Size(3, 3));
    return out;
}

Mat minFilter(const Mat& img) {
    Mat out;
    erode(img, out, Mat());
    return out;
}

Mat boundedSum(const Mat& img1, const Mat& img2) {
    Mat out;
    add(img1, img2, out);
    threshold(out, out, 255, 255, THRESH_TRUNC);
    return out;
}

//----------------------------------------
// 执行图像处理树
//----------------------------------------
Mat executeTree(const shared_ptr<TreeNode>& node, const Mat& input) {
    switch (node->type) {
    case TERMINAL_INPUT:
        return input.clone();
    case MEAN:
        return meanFilter(executeTree(node->children[0], input));
    case MIN:
        return minFilter(executeTree(node->children[0], input));
    case BOUNDED_SUM:
        return boundedSum(executeTree(node->children[0], input), executeTree(node->children[1], input));
    default:
        return input;
    }
}

//----------------------------------------
// 随机生成树节点
//----------------------------------------
random_device rd;
mt19937 rng(rd());
uniform_real_distribution<> prob(0.0, 1.0);

shared_ptr<TreeNode> generateRandomTree(int depth = 0, int maxDepth = 4) {
    if (depth >= maxDepth || prob(rng) < 0.3) {
        return make_shared<TreeNode>(TreeNode{ TERMINAL_INPUT, {} });
    }

    FilterType t = static_cast<FilterType>(1 + (rng() % 3));
    auto node = make_shared<TreeNode>(TreeNode{ t, {} });
    int numChildren = (t == BOUNDED_SUM ? 2 : 1);
    for (int i = 0; i < numChildren; ++i) {
        node->children.push_back(generateRandomTree(depth + 1, maxDepth));
    }
    return node;
}

//----------------------------------------
// 适应度计算
//----------------------------------------
double computeFitness(const Mat& output, const Mat& target) {
    Mat diff;
    absdiff(output, target, diff);
    Scalar s = sum(diff);
    return 1.0 - s[0] / (target.rows * target.cols * 255.0);
}

shared_ptr<TreeNode> cloneTree(const shared_ptr<TreeNode>& node) {
    if (!node) return nullptr;
    auto newNode = make_shared<TreeNode>(TreeNode{ node->type, {} });
    for (auto& child : node->children) {
        newNode->children.push_back(cloneTree(child));
    }
    return newNode;
}

using NodeWithParent = pair<shared_ptr<TreeNode>, shared_ptr<TreeNode>>;

void collectNodesWithParents(const shared_ptr<TreeNode>& node,
    const shared_ptr<TreeNode>& parent,
    vector<NodeWithParent>& result) {
    if (!node) return;
    result.emplace_back(node, parent);
    for (auto& child : node->children) {
        collectNodesWithParents(child, node, result);
    }
}

void crossover(shared_ptr<TreeNode>& a, shared_ptr<TreeNode>& b) {
    vector<NodeWithParent> nodesA, nodesB;
    collectNodesWithParents(a, nullptr, nodesA);
    collectNodesWithParents(b, nullptr, nodesB);

    // 排除根节点（parent 为 nullptr）
    vector<NodeWithParent> validA, validB;
    for (const auto& np : nodesA) {
        if (np.second) validA.push_back(np);
    }
    for (const auto& np : nodesB) {
        if (np.second) validB.push_back(np);
    }

    if (validA.empty() || validB.empty()) return;

    int idxA = rng() % validA.size();
    int idxB = rng() % validB.size();

    auto [nodeA, parentA] = validA[idxA];
    auto [nodeB, parentB] = validB[idxB];

    // 找到 nodeA 在 parentA->children 中的位置
    auto& childrenA = parentA->children;
    auto itA = find(childrenA.begin(), childrenA.end(), nodeA);

    // 找到 nodeB 在 parentB->children 中的位置
    auto& childrenB = parentB->children;
    auto itB = find(childrenB.begin(), childrenB.end(), nodeB);

    // 交换子节点
    if (itA != childrenA.end() && itB != childrenB.end()) {
        swap(*itA, *itB);
    }
}

void mutate(shared_ptr<TreeNode>& node, int maxDepth = 4) {
    vector<shared_ptr<TreeNode>> nodes;
    collectNodes(node, nodes);
    if (nodes.empty()) return;
    int idx = rng() % nodes.size();
    nodes[idx] = generateRandomTree(0, maxDepth);
}

//----------------------------------------
// 主程序（MGG实现）
//----------------------------------------
int main() {
    Mat input = imread("input.png", IMREAD_GRAYSCALE);
    Mat target = imread("target.png", IMREAD_GRAYSCALE);
    if (input.empty() || target.empty()) {
        cerr << "图像读取失败。" << endl;
        return -1;
    }

    const int POP_SIZE = 20;
    const int GENERATIONS = 1000;
    const int OFFSPRING_COUNT = 10;
    const double MUTATION_RATE = 0.9;

    vector<shared_ptr<TreeNode>> population;
    for (int i = 0; i < POP_SIZE; ++i) {
        population.push_back(generateRandomTree());
    }

    shared_ptr<TreeNode> best;
    double bestFitness = -1;

    for (int gen = 0; gen < GENERATIONS; ++gen) {
        // MGG: 随机选两个不同个体作为父代
        int idx1 = rng() % POP_SIZE;
        int idx2 = rng() % POP_SIZE;
        while (idx2 == idx1) idx2 = rng() % POP_SIZE;

        auto parent1 = cloneTree(population[idx1]);
        auto parent2 = cloneTree(population[idx2]);

        vector<pair<double, shared_ptr<TreeNode>>> family;
        family.push_back({ computeFitness(executeTree(parent1, input), target), parent1 });
        family.push_back({ computeFitness(executeTree(parent2, input), target), parent2 });

        // 生成多个子代
        for (int k = 0; k < OFFSPRING_COUNT; ++k) {
            auto childA = cloneTree(parent1);
            auto childB = cloneTree(parent2);
            crossover(childA, childB);
            if (prob(rng) < MUTATION_RATE) mutate(childA);
            if (prob(rng) < MUTATION_RATE) mutate(childB);
            auto chosen = (prob(rng) < 0.5) ? childA : childB;
            double fit = computeFitness(executeTree(chosen, input), target);
            family.push_back({ fit, chosen });
        }

        // 更新最优解
        for (const auto& f : family) {
            if (f.first > bestFitness) {
                bestFitness = f.first;
                best = cloneTree(f.second);
                cout << "[Gen " << gen << "] 最佳适应度: " << bestFitness << endl;
            }
        }

        // 从家庭中选择两个个体（一个最优 + 一个轮盘赌）
        sort(family.rbegin(), family.rend());
        auto elite = family[0];

        // 轮盘赌选择
        double total = 0;
        for (const auto& f : family) total += f.first;
        double r = prob(rng) * total, accum = 0;
        shared_ptr<TreeNode> rouletteSelected = family[1].second; // fallback
        for (const auto& f : family) {
            accum += f.first;
            if (accum >= r) {
                rouletteSelected = f.second;
                break;
            }
        }

        // 替换种群中的两个位置
        population[idx1] = cloneTree(elite.second);
        population[idx2] = cloneTree(rouletteSelected);
    }

    if (best) {
        Mat result = executeTree(best, input);
        imwrite("output.png", result);
        cout << "最佳结果已保存为 output.png" << endl;
    }

    return 0;
}
