import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the student files
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read() for _file in student_files]

# Vectorize the text
def vectorize(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()

def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])

vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()

# Check plagiarism
def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results

# Visualize plagiarism results
def visualize_results(plagiarism_results):
    # Create a matrix for the heatmap
    files = sorted(list(set([pair[0] for pair in plagiarism_results] + [pair[1] for pair in plagiarism_results])))
    matrix = [[0 for _ in files] for _ in files]
    
    file_index = {file: idx for idx, file in enumerate(files)}
    
    for file_a, file_b, score in plagiarism_results:
        i, j = file_index[file_a], file_index[file_b]
        matrix[i][j] = score
        matrix[j][i] = score
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=files, yticklabels=files, cmap='coolwarm', annot=True)
    plt.title('Plagiarism Heatmap')
    plt.show()

# Check for plagiarism and visualize the results
plagiarism_results = check_plagiarism()
visualize_results(plagiarism_results)


