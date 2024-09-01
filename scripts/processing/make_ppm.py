import numpy as np
import csv


# 配列の表示オプションを設定
np.set_printoptions(linewidth=np.inf)

# tsv file description
"""
splice_junction_sample.tsv -> query_data : label row_name splice_junction_sequence
blastn_sample.tsv -> blastn_data : qseqid sseqid qseq sseq qstart qend sstart send mismatch length
"""

# read tsv
with open("../../data/processed/splice_junction_data.tsv", "r", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    query_data = [row for row in reader]

with open("../../data/processed/splice_junction_data.tsv", "r", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    id_list = [row[1] for row in reader]

with open("../../data/processed/blastn_splice_junction_fmt6.tsv", "r", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    blastn_data = [row for row in reader]

# make dictionary of subject sequence by id
def make_sseq_dict(data):
    sseq_by_id = {}
    for i, row in enumerate(data):
        id = row[0]
        sseq = row[3]
        qstart = int(row[4])
        qend = int(row[5])
        sse = [sseq, qstart, qend]
        if id in sseq_by_id:
            sseq_by_id[id].append(sse)
        else:
            sseq_by_id[id] = [sse]
    return sseq_by_id

# definite bases
# bases = {
#     "D": ["A", "G", "T"],
#     "N": ["A", "G", "C", "T"],
#     "S": ["C", "G"],
#     "R": ["A", "G"],
#     "-": ["-"],
#     }
basic_bases = ["A", "C", "G", "T"]
    
# add query sequence information
def add_qseq_vec(matrix, qseq):
    # add query seq information
    for i, base in enumerate(qseq):
        if base in basic_bases:
            matrix[basic_bases.index(base), i] += 1
        elif base == 'D':
            matrix[:, i] += [1/3, 0, 1/3, 1/3]
        elif base == 'N':
            matrix[:, i] += [0.25] * 4
        elif base == 'S':
            matrix[:, i] += [0, 0.5, 0.5, 0]
        elif base == 'R':
            matrix[:, i] += [0.5, 0, 0.5, 0]
    return matrix

# add subject sequence information
def add_sseq_vec(matrix, qseq, sseq, qstart, qend):
    """
    arg : 
        qseq : query seq, sseq : subject seq, qstart : query start position, qend : query end position
    return :
        probability matrix
    """
    # get query seq length
    qlen = len(qseq)
    
    # add query seq information
    for i, base in enumerate(qseq[0:qstart-1], start=0):
        if base in basic_bases:
                matrix[basic_bases.index(base), i] += 1
        elif base == 'D':
            matrix[:, i] += [1/3, 0, 1/3, 1/3]
        elif base == 'N':
            matrix[:, i] += [0.25] * 4
        elif base == 'S':
            matrix[:, i] += [0, 0.5, 0.5, 0]
        elif base == 'R':
            matrix[:, i] += [0.5, 0, 0.5, 0]
    
    # add subject seq information
    for i, base in enumerate(sseq[0:qend-qstart+1], start=qstart-1):
        if base in basic_bases:
                matrix[basic_bases.index(base), i] += 1
        elif base == 'D':
            matrix[:, i] += [1/3, 0, 1/3, 1/3]
        elif base == 'N':
            matrix[:, i] += [0.25] * 4
        elif base == 'S':
            matrix[:, i] += [0, 0.5, 0.5, 0]
        elif base == 'R':
            matrix[:, i] += [0.5, 0, 0.5, 0]
        elif base == '-':
            matrix[:, i] += [0] * 4
            
    # add query seq information
    for i, base in enumerate(qseq[qend:qlen], start=qend):
        if base in basic_bases:
                matrix[basic_bases.index(base), i] += 1
        elif base == 'D':
            matrix[:, i] += [1/3, 0, 1/3, 1/3]
        elif base == 'N':
            matrix[:, i] += [0.25] * 4
        elif base == 'S':
            matrix[:, i] += [0, 0.5, 0.5, 0]
        elif base == 'R':
            matrix[:, i] += [0.5, 0, 0.5, 0]
        
    return matrix

def make_ppm(query_data, id_list, sseq_by_id):
    # 空の行列を作成
    matrices = []
    # id_list から全ての id に対して確率行列を作成 
    for i, id in enumerate(id_list):
        matrix = np.zeros((4, 60))
        qseq = query_data[i][2]
        matrix = add_qseq_vec(matrix, qseq)
        if id in sseq_by_id:
            for sse in sseq_by_id[id]:
                sseq = sse[0]
                qstart = sse[1]
                qend = sse[2]
                matrix = add_sseq_vec(matrix, qseq, sseq, qstart, qend)
            n_hits = len(sseq_by_id[id])
        else:
            n_hits = 0
        
        matrix = matrix / (n_hits + 1)
        matrix = np.around(matrix, decimals=3)
        matrices.append(matrix)
    return matrices
    

def main():
    # key : id, value : sseqのlist　の辞書を作成
    sseq_by_id = make_sseq_dict(blastn_data)
    
    # 確率行列の作成
    matrices = make_ppm(query_data, id_list, sseq_by_id)
    
    # numpyデータの保存
    np.save("../../data/processed/splice_junction_ppm.npy", matrices)

if __name__ == "__main__":
    main()
