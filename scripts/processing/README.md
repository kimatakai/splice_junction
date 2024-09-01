# Splice junction project
2023/09/13

## data file description

- ../../data/processed/blastn_splice_junction_fmt6.tsv

Blastnの出力ファイル。データベースは"RefSeq/H_sapiens"。各カラムは"qseqid sseqid qseq sseq qstart qend sstart send mismatch length"

- ../../data/processed/blastn_splice_junction_fmt10.tsv

Blastnの出力ファイル。データベースは"RefSeq/H_sapiens"。各カラムは"qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"

## program file description

- comma_remove.cpp

tsvファイルから","を取り除くプログラム

- convert_data.cpp

splice_junction.dataを無理矢理tsvファイルに変換するプログラム

- make_fasta.cpp

splice_junction.dataから配列クエリー名と配列情報を取り出して、fastaファイルに変換するプログラム。

- make_pm.py

確率行列を作成するためのPythonファイル。基本的にfmt6のblastn出力ファイルを使用する。
使用するライブラリは"numpy", "csv"

1. データの読み込み
blastnの結果および、splice_junctionの配列データを読み込む。データはリストとして保管しておく。
query_data : クエリー配列
id_list : ID
blastn_data : blastnの出力結果（行列データ）

2. 関数定義
make_sseq_dict() : IDをキー、["マルチプルアラインメント配列", 開始位置, 終了位置]のリストをバリューとする辞書を作成する関数
add_qseq_vec() : 引数には行列と、クエリー配列。行列にクエリー配列のone-hotベクトル足す関数
add_sseq_vec() : 引数には行列と、クエリー配列、マルチプルアラインメント配列とその情報。行列に類似配列のベクトルを足す関数。類似部分以外はクエリー配列の情報を足し合わせる。
make_ppm() : add_qseq_vec()とadd_sseq_vec()関数を用いて、確率行列を作成する関数。

3. メイン関数
出力として、"../../data/processed/"に.npyファイルとして、確率行列を保存する。