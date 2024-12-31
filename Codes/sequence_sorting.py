fasta_content="D:\Repo\A_Universal_Framework_For_Clusetering_Sequences\Datasets\Ajinkya_Synthetic_Dataset\AJR_Dataset_Sequences_100k.fa"

# Preprocessing: Extracting and encoding sequences
def parse_fasta(content):
    sequences = []
    labels = []
    with open(content, 'r') as file:
        for line in file:
            if line.startswith("#"):
                label = line[-3:]  # Assuming label is the last character of the line
                label=label.strip()
                labels.append(label)
            else:
                sequences.append(line.strip())
    
    return sequences, labels

# Extract sequences from the file
sequences,labels= parse_fasta(fasta_content)

# fasta_dict={'1':[],'0':[]}
fasta_dict={'0A':[],'0T': [],'0G': [],'0C': [], '1A': [],'1T': [],'1G': [],'1C': []}

# print(sequences)
for i in range(len(sequences)):
    # print(sequences[i],labels[i])
    fasta_dict[str(labels[i])].append(str(sequences[i]))


# print(fasta_dict)
j=0
with open(f'D:\Repo\A_Universal_Framework_For_Clusetering_Sequences\Datasets\Sorted_Sequences\AJR_Dataset_Sequences_100k_Sorted.fa','w') as file:
    for key in fasta_dict:
            seq=fasta_dict[key]
            for i in range(len(seq)):
                file.write(f"#{j}_{key}\n")
                j+=1
                file.write(seq[i]+"\n")

print("Sequence Sorted")