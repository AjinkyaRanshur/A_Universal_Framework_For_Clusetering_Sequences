import random

# Mapping nucleotides to integers
seqtoint = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
nucleotide = ["A", "T", "G", "C"]

# # Transition probability matrix
# transition_matrix_1 = [
#     [0.9, 0.03, 0.04, 0.03],  # A -> [A, T, G, C]
#     [0.03, 0.9, 0.03, 0.04],  # T -> [A, T, G, C]
#     [0.04, 0.03, 0.9, 0.03],  # G -> [A, T, G, C]
#     [0.03, 0.04, 0.03, 0.9],  # C -> [A, T, G, C]
# ]

# transition_matrix_2 = [
#     [0.05, 0.9, 0.03, 0.02],  # A -> [A, T, G, C]
#     [0.9, 0.05, 0.02, 0.03],  # T -> [A, T, G, C]
#     [0.03, 0.02, 0.05, 0.9],  # G -> [A, T, G, C]
#     [0.02, 0.03, 0.9, 0.05],  # C -> [A, T, G, C]
# ]

# Transition probability matrix of CpG Island
transition_matrix_1 = [
    [0.180, 0.120, 0.426, 0.274],  # A -> [A, T, G, C]
    [0.079, 0.182, 0.384, 0.355],  # T -> [A, T, G, C]
    [0.161, 0.125, 0.373, 0.339],  # G -> [A, T, G, C]
    [0.171, 0.188, 0.274, 0.368],  # C -> [A, T, G, C]
]

#Non CpG Island Matrix
transition_matrix_2 = [
    [0.300, 0.210, 0.285, 0.205],  # A -> [A, T, G, C]
    [0.177, 0.292, 0.292, 0.239],  # T -> [A, T, G, C]
    [0.248, 0.208, 0.298, 0.246],  # G -> [A, T, G, C]
    [0.322, 0.302, 0.078, 0.298],  # C -> [A, T, G, C]
]


# Function to get the next nucleotide based on maximum probability
def next_nucleotide_max(current_nucleotide,choosemat):
    current_index = seqtoint[current_nucleotide]
    if choosemat==1:
        probabilities = transition_matrix_1[current_index]
    else:
        probabilities = transition_matrix_2[current_index]
    # Use random.choices to select the next nucleotide probabilistically
    return random.choices(nucleotide, weights=probabilities, k=1)[0]

# Function to initialize the first nucleotide and sequence length
def initialize_sequence():
    first_nucleotide = random.choice(nucleotide)
    sequence_length = random.randint(32, 100)
    return first_nucleotide, sequence_length

# Generate 100 genomic sequences
with open("D:\Repo\A_Universal_Framework_For_Clusetering_Sequences\Datasets\CPG_Island\Cpg_Dataset_Sequences.fa",'w') as f:
    # genomic_sequences = []
    for i in range(100000):
        first_nucleotide, sequence_length = initialize_sequence()
        sequence = first_nucleotide  # Start the sequence with the first nucleotide
        choosemat=random.randint(0,1)
        # save_name=str(choosemat)+sequence
        save_name=str(choosemat)
        # Build the sequence by iteratively adding the next nucleotide
        for _ in range(sequence_length - 1):
            next_nuc = next_nucleotide_max(sequence[-1],choosemat)
            sequence += next_nuc

        f.write(f"#{i}_{save_name}\n")
        f.write(sequence+"\n")

print("Sequences Generated")
