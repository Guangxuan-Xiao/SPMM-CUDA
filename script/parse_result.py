import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()
with open(args.input, 'r') as f:
    lines = f.readlines()
total_len = len(lines)
dset_line = 0
kLen_line = 2
cusparse_line = 9
ours_line = 12
incre = 19
fout = open(args.output, 'w+')
print("dset, kLen, cusparse, ours, speedup, improve", file=fout)
while (dset_line < total_len):
    dset = lines[dset_line].split(' ')[4].strip('"')
    kLen = int(lines[kLen_line].split(' ')[4])  
    if dset == 'citation' and kLen == 512:
        incre = 10
    else:
        cusparse = float(lines[cusparse_line].split(' ')[6])
        ours = float(lines[ours_line].split(' ')[6])
        print(f"{dset}, {kLen}, {cusparse}, {ours}, {cusparse/ours}, {cusparse/ours > 1}", file=fout)
        incre = 19
    dset_line += incre
    kLen_line += incre
    cusparse_line += incre
    ours_line += incre
fout.close()