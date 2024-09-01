/* header file include */
#include <iostream>     // 標準入出力ストリームにアクセスするため
#include <fstream>      // ファイル入出力に関連する機能を使用するため
#include <string>       // 文字列操作に必要な機能を使用するため
#include <sstream>      // 文字列ストリーム操作（std::istringstream など）に必要

int main() {
    std::ifstream infile("splice.data");  // Input file // std = standard library
    std::ofstream outfile("splice_junction_data.tsv");  // Output file in tab-separated format
    std::string line;   // 文字列'line'を作成（空）

    // std::getline(infile, line) で入力ファイルから一行ずつ読み込み、それを line に格納。ファイルの終わりに達するまでループ。
    while (std::getline(infile, line)) {
        std::istringstream iss(line);       // istringstreamオブジェクトissを作成   // 文字列'line'の内容をstreamとして扱う // 空白文字で区切られる
        std::string type, id, sequence;     // 3つの std::string 型の変数（type, id, sequence）を宣言   // 後に、iss ストリームから値を読み取るために使用

        // Parse the line
        iss >> type >> id >> sequence;      // iss ストリームからデータを読み取り、それぞれの変数（type, id, sequence）に代入

        // Remove trailing comma from 'type' and 'id'
        if (type.back() == ',') {       // back() というメンバ関数, 文字列の最後の文字を返す
            type.pop_back();        // remove ','
        }
        if (id.back() == ',') {
            id.pop_back();
        }

        // Write to the tab-separated file
        outfile << type << "\t" << id << "\t" << sequence << std::endl;
    }

    infile.close();
    outfile.close();

    return 0;
}