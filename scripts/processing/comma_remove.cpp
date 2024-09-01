#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

int main() {
    std::ifstream infile("blastn_splice_junction.txt");  // 入力ファイルをオープン
    std::ofstream outfile("blastn_splice_junction.tsv");  // 出力ファイルをオープン（タブ区切り）

    std::string line;  // 各行を格納する文字列

    // 入力ファイルから一行ずつ読み込む
    while (std::getline(infile, line)) {
        std::istringstream iss(line);  // istringstreamオブジェクトを作成
        std::string token;  // 各トークン（値）を格納する文字列
        std::vector<std::string> tokens;  // すべてのトークンを格納する動的配列

        // スペースやタブで区切られた各トークンを読み込む
        while (iss >> token) {
            tokens.push_back(token);  // トークンを動的配列に追加
        }

        // 最初のトークン（type）の末尾がカンマであれば、それを取り除く
        if (tokens[0].back() == ',') {
            tokens[0].pop_back();
        }

        // タブで区切って出力ファイルに書き込む
        for (size_t i = 0; i < tokens.size(); ++i) {
            outfile << tokens[i];
            if (i < tokens.size() - 1) {
                outfile << "\t";
            }
        }
        outfile << std::endl;  // 行の終わりに改行を追加
    }

    // ファイルストリームを閉じる
    infile.close();
    outfile.close();

    return 0;
}

